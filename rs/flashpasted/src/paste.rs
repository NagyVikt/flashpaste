//! End-to-end paste dispatch — the hot path.
//!
//! Sequence (mirrors the bash dispatcher's FAST PATH block, modulo:
//! clipboard ownership is already done by the daemon, so we can skip every
//! probe/prestage step):
//!
//!   1. `tmux select-pane -t <pane>`
//!   2. `tmux unbind -n C-v`   (fact #2 — must happen BEFORE send-text)
//!   3. kitty IPC: `send_text` with payload `\x16`  (fact #1)
//!   4. schedule detached `tmux bind -n C-v ...` after 100ms (fact #2)
//!
//! Steps 2 and 3 must run in that order. We `select-pane` first because the
//! caller's tmux paste menu may be on a different pane than the focused
//! one; without selecting we'd send-text into the wrong pane.

use std::sync::Arc;

use anyhow::{Context, Result};
use tracing::info;

use crate::agent::{self, AgentKind};
use crate::state::{now_unix_ms, SharedState, StagedImage, StagedText};
use crate::tmux;
use tokio::io::AsyncWriteExt;

/// Top-level entry from `ipc::handle_paste`.
///
/// `pane` is the tmux pane id (e.g. `%4`).
/// `_staged` is included so the caller can confirm the image is fresh
/// before we burn time on the IPC + tmux dance.
pub async fn dispatch_image_paste(
    state: Arc<SharedState>,
    pane: String,
    staged: StagedImage,
) -> Result<()> {
    let payload_bytes = staged.bytes.len();
    let payload_name = staged
        .path
        .file_name()
        .and_then(|n| n.to_str())
        .map(|s| s.trim_start_matches("Screenshot from ").to_string())
        .unwrap_or_else(|| "<no-name>".to_string());

    // Re-claim X11 CLIPBOARD with the staged image bytes so Claude's
    // `wl-paste -t image/png` reads OUR image (and not whatever external
    // app last grabbed the selection). Wayland is mutter-wedged on this
    // box; the X11 owner does the real work.
    //
    // No sleep: the X11 reclaim wakes on the notifier and runs concurrently
    // with the rest of the dispatch (tmux forks, send-keys). By the time
    // Claude actually fires `wl-paste -t image/png`, the reclaim has long
    // since landed — local X11 socket round-trips are sub-millisecond,
    // while the tmux forks below add several ms of scheduling. If a race
    // ever shows up as "no image found" on the first paste after a new
    // screenshot, restore a 2 ms sleep here.
    let _ = state.stage_notifier_tx.send(now_unix_ms());

    let snap = tmux::pane_snapshot(&pane).await;
    let agent = agent::detect_cached(&state.agent_cache, &pane, &snap).await;

    // Experimental (opt-in via FLASHPASTE_IMAGE_AS_PATH=1): instead of the
    // X11-reclaim + Ctrl-V + clipboard-read dance, write the staged image to
    // a stable file and inject its PATH through the fast text path. Claude
    // Code / Codex can attach an image referenced by path, turning a ~38 ms
    // image paste into a ~text-speed one and skipping the clipboard entirely.
    //
    // Off by default: whether the TUI auto-attaches from a bare pasted path
    // is version-dependent, so this is a flag the user flips on if it works
    // for their build rather than a behaviour change everyone inherits.
    if image_as_path_enabled() && matches!(agent, AgentKind::ClaudeCode | AgentKind::Codex) {
        let image_path = agent::materialize_readable_path(&staged)
            .await
            .context("materialize image path for path-paste")?;
        let mut line = std::os::unix::ffi::OsStrExt::as_bytes(image_path.as_os_str()).to_vec();
        line.push(b' ');
        let text = StagedText {
            bytes: Arc::new(line),
            captured_at: std::time::SystemTime::now(),
        };
        dispatch_text_paste(state.clone(), pane.clone(), text).await?;
        info!(
            pane,
            kind = "image",
            agent = agent.as_str(),
            payload_bytes,
            payload_name = %payload_name,
            path = %image_path.display(),
            "PASTED image as path (FLASHPASTE_IMAGE_AS_PATH)"
        );
        return Ok(());
    }

    if agent == AgentKind::Aider {
        let image_path = agent::deliver_aider_image(&pane, &staged)
            .await
            .context("aider image delivery")?;
        info!(
            pane,
            kind = "image",
            agent = agent.as_str(),
            payload_bytes,
            payload_name = %payload_name,
            path = %image_path.display(),
            "PASTED image via agent adapter"
        );
        return Ok(());
    }

    // Inject Ctrl-V (0x16) into the pane's pty via `tmux send-keys -l`.
    // `-l` is literal: no keytable, no unbind/rebind dance. Reaches any
    // tmux pane regardless of which terminal hosts the client.
    tmux::dispatch_ctrl_v_to_pane(&pane)
        .await
        .context("batched tmux Ctrl-V dispatch")?;

    info!(
        pane,
        kind = "image",
        agent = agent.as_str(),
        payload_bytes,
        payload_name = %payload_name,
        "PASTED image"
    );
    Ok(())
}

/// Text-paste dispatch. Pipes the staged text bytes into a tmux buffer
/// via `tmux load-buffer -` (stdin), then `tmux paste-buffer -p -t <pane>`
/// writes the buffer bytes directly into the target pane's pty. No
/// clipboard claim, no kitty IPC, no unbind/rebind dance — just two
/// `tmux` forks and Claude Code reads the text as if the user typed it.
///
/// Works for ANY tmux pane regardless of which terminal hosts the tmux
/// client (same property as `tmux send-keys -l ^V` for image paste).
/// User contract (2026-05-19): "if last time was text and no new
/// screenshot was taken, text should be pasted to each terminal" — this
/// is what makes that contract hold across multiple panes.
pub async fn dispatch_text_paste(
    _state: Arc<SharedState>,
    pane: String,
    text: StagedText,
) -> Result<()> {
    let bytes_len = text.bytes.len();
    let sanitized = sanitize_clipboard_text(&text.bytes);

    // ─── Last-gate outcome guard ──────────────────────────────────────
    // The daemon must NEVER write image/binary bytes into a pane as text.
    // Every staging path *should* have rejected non-text already, but this
    // is the single chokepoint all text pastes flow through, so guard here
    // too: if the bytes we're about to paste aren't text, refuse and log a
    // visible outcome instead of dumping a wall of `\x89PNG…` into the
    // prompt. Cheap (a magic/NUL/UTF-8 check on the leading bytes).
    if !crate::ipc::looks_like_text(sanitized.as_ref()) {
        tracing::warn!(
            pane,
            kind = "text",
            bytes = bytes_len,
            outcome = "rejected-nontext",
            "REFUSED text paste: payload is image/binary, not text (leak blocked at last gate)"
        );
        anyhow::bail!("refusing to paste non-text payload as text");
    }
    let html_sanitized = matches!(sanitized, std::borrow::Cow::Owned(_));

    // Single tmux fork: `load-buffer - ; paste-buffer` chained with tmux's
    // `;` command separator. tmux runs both commands in one process — the
    // load reads our text from stdin, then the paste writes the buffer into
    // the target pane. Halves the fork count of the hot text path (was two
    // separate `tmux` spawns). `-p` keeps bracketed paste for multi-line
    // safety. Passing `;` as its own argv entry (no shell) is how tmux sees
    // a command separator when exec'd directly.
    let mut child = tokio::process::Command::new("tmux")
        .args([
            "load-buffer",
            "-b",
            "fp_text",
            "-",
            ";",
            "paste-buffer",
            "-p",
            "-b",
            "fp_text",
            "-t",
            &pane,
        ])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .context("spawn tmux load-buffer;paste-buffer")?;
    {
        let mut stdin = child.stdin.take().context("tmux stdin not piped")?;
        stdin
            .write_all(sanitized.as_ref())
            .await
            .context("write tmux stdin")?;
    }
    let status = child.wait().await.context("tmux load;paste wait")?;
    if !status.success() {
        anyhow::bail!("tmux load-buffer;paste-buffer non-zero: {:?}", status);
    }

    info!(
        pane,
        kind = "text",
        bytes = bytes_len,
        sent_bytes = sanitized.as_ref().len(),
        html_sanitized,
        outcome = "clean",
        "PASTED text"
    );
    Ok(())
}

/// Strip HTML markup from clipboard text so that content copied from
/// browsers (Facebook, product pages, social media) pastes as clean
/// plain text instead of raw `<span>`, `<div>`, `<img blob:…/>` etc.
///
/// Only fires when the text contains an HTML-ish tag opener (`<letter`
/// or `</`). Plain text, code with `<`/`>` operators, and non-UTF-8
/// bytes all pass through untouched.
fn sanitize_clipboard_text(bytes: &[u8]) -> std::borrow::Cow<'_, [u8]> {
    let Ok(text) = std::str::from_utf8(bytes) else {
        return std::borrow::Cow::Borrowed(bytes);
    };
    if !looks_like_html(text) {
        return std::borrow::Cow::Borrowed(bytes);
    }
    let out = html_to_plaintext(text);
    // Safety net: if stripping tags produced nothing but whitespace, the input
    // was not prose-bearing HTML (it was markup-shaped data — XML/code/config
    // whose payload lives inside the tags/attributes). Pasting an empty string
    // is the worst outcome, so keep the original bytes untouched.
    if out.trim().is_empty() && !text.trim().is_empty() {
        return std::borrow::Cow::Borrowed(bytes);
    }
    if out.as_bytes() == bytes {
        std::borrow::Cow::Borrowed(bytes)
    } else {
        std::borrow::Cow::Owned(out.into_bytes())
    }
}

/// Returns true only when `text` carries genuine HTML markup that should be
/// flattened to plain text (i.e. content copied from a rendered web page).
///
/// The old test — any `<` followed by a letter or `/` — was far too broad: it
/// fired on XML, source code, and config files that merely contain tag-shaped
/// syntax. A URDF (`<robot>`, `<xacro:property/>`, `<link>`) is almost
/// entirely such tags, so it was classified as HTML and `html_to_plaintext`
/// stripped it to nothing — the "copied file pastes empty" bug. We now require
/// a recognised HTML element (matched with a tag boundary so `<li>` does not
/// fire on `<link>`, nor `<b>` on `<box>`), an HTML closing tag, an HTML
/// entity, or a doctype. When unsure, treat it as plain text and leave it
/// untouched — losing a bit of HTML cleanup is far better than destroying a
/// user's code/XML paste.
fn looks_like_html(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();

    // Entities only occur in HTML.
    const ENTITIES: &[&str] = &[
        "&nbsp;", "&amp;", "&lt;", "&gt;", "&quot;", "&copy;", "&euro;", "&#",
    ];
    if ENTITIES.iter().any(|e| lower.contains(e)) {
        return true;
    }
    if lower.contains("<!doctype html") {
        return true;
    }

    // Recognised HTML element names. Matched as `<name`/`</name` with a tag
    // boundary after the name so an XML element that merely shares a prefix
    // (e.g. `<link>`, `<box>`, `<parent>`) does not register.
    const TAGS: &[&str] = &[
        "html",
        "head",
        "body",
        "div",
        "span",
        "p",
        "br",
        "hr",
        "a",
        "img",
        "ul",
        "ol",
        "li",
        "table",
        "thead",
        "tbody",
        "tr",
        "td",
        "th",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "b",
        "i",
        "u",
        "em",
        "strong",
        "small",
        "pre",
        "code",
        "blockquote",
        "font",
        "style",
        "script",
        "button",
        "label",
        "nav",
        "header",
        "footer",
        "section",
        "article",
    ];
    let bytes = lower.as_bytes();
    TAGS.iter()
        .any(|tag| has_tag_open(bytes, tag, false) || has_tag_open(bytes, tag, true))
}

/// True when `haystack` contains `<name` (or `</name` when `closing`) followed
/// by a tag-boundary byte. The boundary check is what prevents `li` matching
/// `<link>` or `b` matching `<box>`.
fn has_tag_open(haystack: &[u8], name: &str, closing: bool) -> bool {
    let needle = if closing {
        format!("</{name}")
    } else {
        format!("<{name}")
    };
    let nb = needle.as_bytes();
    haystack.windows(nb.len() + 1).any(|w| {
        &w[..nb.len()] == nb && matches!(w[nb.len()], b' ' | b'>' | b'/' | b'\n' | b'\r' | b'\t')
    })
}

/// Convert HTML to plain text:
///   - `<br>` / `<br/>` → newline
///   - block elements (`<p>`, `<div>`, headings, `<li>`, `<tr>`) → newline
///   - `<img src="blob:…">` → `[domain image]` label (preserves context)
///   - all other tags stripped
///   - common HTML entities decoded (`&amp;`, `&nbsp;`, `&lt;`, …)
///   - runs of 3+ blank lines collapsed to 2
fn html_to_plaintext(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;

    while let Some(lt) = rest.find('<') {
        out.push_str(&rest[..lt]);
        let from_lt = &rest[lt..];

        let Some(gt_off) = from_lt.find('>') else {
            // Unclosed tag — emit literally and stop.
            out.push_str(from_lt);
            rest = "";
            break;
        };

        let inner = &from_lt[1..gt_off];
        let tag_end = lt + gt_off + 1;

        // Tag name: lowercase, trailing `/` stripped (handles `<br/>`).
        let raw = inner.trim_start();
        let name = raw
            .split_ascii_whitespace()
            .next()
            .unwrap_or("")
            .to_ascii_lowercase();
        let name = name.trim_end_matches('/');

        match name {
            "br" => out.push('\n'),
            // Block-level elements: ensure a newline boundary.
            "p" | "/p" | "div" | "/div" | "section" | "/section" | "article" | "/article"
            | "li" | "tr" | "/tr" | "h1" | "h2" | "h3" | "h4" | "h5" | "h6" | "/h1" | "/h2"
            | "/h3" | "/h4" | "/h5" | "/h6" => {
                if !out.ends_with('\n') && !out.is_empty() {
                    out.push('\n');
                }
            }
            // Blob-URL images keep a short human-readable label.
            "img" if inner.contains("blob:") => {
                out.push_str(&blob_img_label(inner));
            }
            // Everything else (span, a, b, strong, em, img without blob:,
            // script, style …) is silently dropped.
            _ => {}
        }

        rest = &rest[tag_end..];
    }

    out.push_str(rest);
    let out = decode_html_entities(out);
    collapse_blank_lines(out)
}

/// Decode common named and numeric HTML entities.
fn decode_html_entities(mut text: String) -> String {
    if !text.contains('&') {
        return text;
    }
    const NAMED: &[(&str, &str)] = &[
        ("&amp;", "&"),
        ("&lt;", "<"),
        ("&gt;", ">"),
        ("&nbsp;", " "),
        ("&apos;", "'"),
        ("&quot;", "\""),
        ("&mdash;", "—"),
        ("&ndash;", "–"),
        ("&hellip;", "…"),
        ("&laquo;", "«"),
        ("&raquo;", "»"),
        ("&#39;", "'"),
        ("&#160;", " "),
    ];
    for (entity, rep) in NAMED {
        if text.contains(entity) {
            text = text.replace(entity, rep);
        }
    }
    text
}

/// Collapse runs of 3+ consecutive newlines to 2 and trim leading/trailing
/// whitespace — removes the blank-line padding that HTML layout adds.
fn collapse_blank_lines(text: String) -> String {
    let mut out = String::with_capacity(text.len());
    let mut nl_run: u8 = 0;
    for ch in text.chars() {
        if ch == '\n' {
            nl_run = nl_run.saturating_add(1);
            if nl_run <= 2 {
                out.push('\n');
            }
        } else {
            nl_run = 0;
            out.push(ch);
        }
    }
    out.trim().to_string()
}

/// Build a short label for a blob-URL `<img>` tag, e.g. `[facebook.com image]`.
fn blob_img_label(tag: &str) -> String {
    extract_blob_domain(tag)
        .map(|d| format!("[{d} image]"))
        .unwrap_or_else(|| "[image]".to_string())
}

/// Extract the hostname from a `blob:https://hostname/...` URL inside a tag.
fn extract_blob_domain(tag: &str) -> Option<String> {
    let after_blob = tag.split_once("blob:")?.1;
    let after_scheme = after_blob
        .strip_prefix("https://")
        .or_else(|| after_blob.strip_prefix("http://"))
        .unwrap_or(after_blob);
    let end = after_scheme
        .find(['/', '"', '\'', ' '])
        .unwrap_or(after_scheme.len());
    let domain = &after_scheme[..end];
    if domain.is_empty() {
        return None;
    }
    Some(domain.strip_prefix("www.").unwrap_or(domain).to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_text_borrowed_unchanged() {
        let input = b"just plain text with no tags";
        assert!(matches!(
            sanitize_clipboard_text(input),
            std::borrow::Cow::Borrowed(_)
        ));
    }

    #[test]
    fn less_than_in_code_not_treated_as_html() {
        // `<` followed by a digit or space is not a tag opener.
        let input = b"if x < 5 and y > 3 then z";
        let out = sanitize_clipboard_text(input);
        assert_eq!(out.as_ref(), input);
    }

    #[test]
    fn non_utf8_bytes_passed_through() {
        let input: &[u8] = &[0xff, 0xfe, b'<', b'i', b'm', b'g'];
        let out = sanitize_clipboard_text(input);
        assert_eq!(out.as_ref(), input);
    }

    #[test]
    fn urdf_xml_not_treated_as_html() {
        // Regression: a URDF is almost entirely tags whose payload lives in
        // attributes. The old `<`+letter test classified it as HTML and
        // stripped it to nothing. It must now pass through byte-for-byte.
        let input = br#"<?xml version="1.0" encoding="utf-8"?>
<robot name="atirobot" xmlns:xacro="http://ros.org/wiki/xacro">
  <xacro:property name="base_width" value="0.45"/>
  <link name="base_link"/>
  <joint name="base_to_wheel" type="continuous"/>
</robot>"#;
        let out = sanitize_clipboard_text(input);
        assert_eq!(out.as_ref(), input, "URDF/XML must not be HTML-stripped");
    }

    #[test]
    fn cyclonedds_config_xml_not_treated_as_html() {
        let input = br#"<CycloneDDS xmlns="https://cdds.io/config">
  <Domain id="any">
    <General><AllowMulticast>false</AllowMulticast></General>
  </Domain>
</CycloneDDS>"#;
        let out = sanitize_clipboard_text(input);
        assert_eq!(out.as_ref(), input);
    }

    #[test]
    fn html_sharing_a_prefix_with_xml_still_detected() {
        // `<link>` (XML) must not trigger, but real `<li>`/`<div>` must.
        assert!(!looks_like_html("<link name=\"base\"/>"));
        assert!(!looks_like_html("<box size=\"1 2 3\"/>"));
        assert!(!looks_like_html("<parent link=\"a\"/>"));
        assert!(looks_like_html("<ul><li>one</li><li>two</li></ul>"));
        assert!(looks_like_html("<div>hi</div>"));
    }

    #[test]
    fn all_tag_html_does_not_strip_to_empty() {
        // Even if something all-tags slips past looks_like_html, the safety
        // net keeps the original rather than pasting an empty string.
        let input = b"<custom:thing a=\"1\"/><custom:thing b=\"2\"/>";
        let out = sanitize_clipboard_text(input);
        assert_eq!(out.as_ref(), input);
    }

    #[test]
    fn blob_img_replaced_with_domain_label() {
        let input = b"check this <img src=\"blob:https://www.facebook.com/c4a1d5d6\"/> nice";
        let out = sanitize_clipboard_text(input);
        assert_eq!(
            std::str::from_utf8(out.as_ref()).unwrap(),
            "check this [facebook.com image] nice"
        );
    }

    #[test]
    fn multiple_blob_imgs_all_replaced() {
        let input = b"a <img src=\"blob:https://www.facebook.com/aaa\"/> b <img src=\"blob:https://www.facebook.com/bbb\"/> c";
        let out = sanitize_clipboard_text(input);
        assert_eq!(
            std::str::from_utf8(out.as_ref()).unwrap(),
            "a [facebook.com image] b [facebook.com image] c"
        );
    }

    #[test]
    fn regular_img_tag_stripped() {
        // A non-blob <img> has no meaning outside the browser — strip it.
        let input = b"before <img src=\"https://example.com/x.jpg\"/> after";
        let out = sanitize_clipboard_text(input);
        assert_eq!(std::str::from_utf8(out.as_ref()).unwrap(), "before  after");
    }

    #[test]
    fn span_and_inline_tags_stripped() {
        let input = b"<span class=\"price\"><b>1 800 \xe2\x82\xac</b></span>";
        let out = sanitize_clipboard_text(input);
        assert_eq!(std::str::from_utf8(out.as_ref()).unwrap(), "1 800 €");
    }

    #[test]
    fn br_becomes_newline() {
        let input = b"line one<br/>line two<br />line three";
        let out = sanitize_clipboard_text(input);
        assert_eq!(
            std::str::from_utf8(out.as_ref()).unwrap(),
            "line one\nline two\nline three"
        );
    }

    #[test]
    fn block_elements_add_newlines() {
        let input = b"<div>first</div><div>second</div>";
        let out = sanitize_clipboard_text(input);
        assert_eq!(std::str::from_utf8(out.as_ref()).unwrap(), "first\nsecond");
    }

    #[test]
    fn html_entities_decoded() {
        // Entities only appear inside HTML; the surrounding tag triggers the
        // HTML path, which then decodes them.
        let input = b"<p>price &amp; quality &lt;= great &gt; average&nbsp;yes</p>";
        let out = sanitize_clipboard_text(input);
        assert_eq!(
            std::str::from_utf8(out.as_ref()).unwrap(),
            "price & quality <= great > average yes"
        );
    }

    #[test]
    fn text_path_rejects_image_bytes() {
        // The last-gate guard: PNG bytes that somehow reached the text path
        // must be refused, not pasted as a wall of garbage.
        let png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x01\x00";
        let sanitized = sanitize_clipboard_text(png);
        assert!(!crate::ipc::looks_like_text(sanitized.as_ref()));
    }

    #[test]
    fn text_path_accepts_real_text() {
        let t = b"see https://example.com/p?x=1 for details";
        let sanitized = sanitize_clipboard_text(t);
        assert!(crate::ipc::looks_like_text(sanitized.as_ref()));
    }

    #[test]
    fn facebook_product_page_roundtrip() {
        // Realistic slice of what Facebook copies for a product listing.
        let input = b"<div><span>Lateral Raise</span><br/><img src=\"blob:https://www.facebook.com/uuid\"/><p>261 kg &amp; 1 800 &euro;</p></div>";
        let out = sanitize_clipboard_text(input);
        let s = std::str::from_utf8(out.as_ref()).unwrap();
        assert!(s.contains("Lateral Raise"));
        assert!(s.contains("[facebook.com image]"));
        assert!(s.contains("261 kg & 1 800"));
        assert!(!s.contains("<"));
        assert!(!s.contains("blob:"));
    }
}

/// Opt-in switch for the image-as-path fast path. Reads `FLASHPASTE_IMAGE_AS_PATH`
/// once per paste — cheap, and lets the user toggle it without restarting if
/// the daemon is launched through a wrapper that re-reads the env. Truthy
/// values: `1`, `true`, `yes`, `on` (case-insensitive).
fn image_as_path_enabled() -> bool {
    std::env::var("FLASHPASTE_IMAGE_AS_PATH")
        .ok()
        .map(|v| {
            let v = v.trim().to_ascii_lowercase();
            matches!(v.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}
