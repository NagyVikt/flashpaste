//! Unix-socket IPC. `flashpaste-trigger` connects, sends a length-prefixed
//! JSON request, receives a length-prefixed JSON response.
//!
//! Wire format (matches `flashpaste-trigger/src/main.rs`):
//!   request:  4-byte LE u32 length || JSON bytes
//!   response: 4-byte LE u32 length || JSON bytes
//!
//! Request body:
//!   {"op":"paste","pane":"%4","ts":"..."}
//!   {"op":"stage","image_path":"..."}    // Phase 3 hook
//!
//! Response bodies:
//!   {"ok":true,"latency_ms":7}
//!   {"ok":true,"deduped":true}
//!   {"ok":false,"reason":"no-image","fallback":"bash"}

use std::io::Read;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};
use tokio::task::JoinHandle;
use tracing::{debug, error, info, warn};

use crate::paste;
use crate::state::{now_unix_ms, SharedState, StagedImage, StagedSelection, StagedText};

/// Cap incoming request size. 16KB was enough when `Stage` only carried a
/// path; `StageText` inlines the bytes (base64-encoded by the trigger) so
/// we widen to 8MB — covers a 6MB clipboard payload comfortably. Anything
/// larger should hit the daemon via a path field, not inline.
const MAX_REQUEST_BYTES: u32 = 8 * 1024 * 1024;
/// How long a paste with an identical `(pane, content)` signature is
/// suppressed after the last trigger carrying that signature — accepted or
/// absorbed (see `dedup_decide`; the window slides). Sized to swallow the
/// dual-handler
/// spread: kitty's `map ctrl+v` launches the paste router as a BACKGROUND
/// process, so its redundant fire can land well after tmux's `bind -n C-v`
/// already pasted — a real measurement on this box showed the two pastes
/// **1.84 s apart** (12:16:31.417 then :33.253, identical 972-byte payload),
/// which sailed past the old 1 s window and produced a visible double. The
/// guard keys on pane+content (see `paste_signature`, which also trims
/// trailing newlines), so this wider window never blocks a genuinely
/// different paste — only a verbatim re-fire of the same bytes into the same
/// pane within ~2.5 s, which is exactly the bug.
const PASTE_DEDUP_WINDOW_MS: u64 = 2500;
/// Same guard, but for IMAGE pastes, where the window has to be wider.
///
/// Two reasons the text window is too tight here. An image dispatch takes
/// ~450 ms end-to-end (X11 re-claim + tmux fork + the agent's own clipboard
/// read) against ~3 ms for text, so there is far more dead time in which the
/// paste "feels stuck" and the user presses again. And a re-press is spaced by
/// human reaction, not by machine latency: a replay of the measured burst on
/// this box landed its second press 2.54 s after the first — 40 ms past the
/// text window — and pasted the same screenshot twice even with the window
/// sliding. 5 s covers a mash at human cadence.
///
/// Safe to widen because the guard is keyed on `(pane, content)`: only a
/// verbatim re-fire of the SAME image into the SAME pane is suppressed.
/// Deliberately attaching one screenshot twice still works — it just needs
/// 5 s of quiet, which is far cheaper than silently double-attaching a 250 KB
/// image to a chat prompt and paying for it twice.
const PASTE_DEDUP_WINDOW_IMAGE_MS: u64 = 5000;

#[derive(Debug, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum Request {
    Paste {
        pane: String,
        #[allow(dead_code)]
        ts: Option<String>,
    },
    Stage {
        image_path: String,
    },
    /// v1.19+: stage raw bytes as a text selection. Payload is base64 so
    /// the JSON envelope is safe for binary text (tmux occasionally pipes
    /// non-UTF8). `from` is informational (e.g. "tmux:%21").
    StageText {
        bytes_b64: String,
        #[allow(dead_code)]
        from: Option<String>,
    },
    Ping,
}

pub async fn spawn_listener(state: Arc<SharedState>) -> Result<JoinHandle<()>> {
    let socket_path = state.config.socket_path.clone();

    // Remove a stale socket from a previous run. We don't `unlink` files
    // that aren't sockets — that'd be how a misconfigured `--socket` flag
    // accidentally nukes real data.
    if socket_path.exists() {
        if is_socket(&socket_path) {
            let _ = std::fs::remove_file(&socket_path);
        } else {
            anyhow::bail!(
                "socket path {} exists but is not a socket — refusing to overwrite",
                socket_path.display()
            );
        }
    }
    // Ensure the parent dir is there.
    if let Some(parent) = socket_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create socket parent {}", parent.display()))?;
    }

    let listener = UnixListener::bind(&socket_path)
        .with_context(|| format!("bind {}", socket_path.display()))?;
    // 0600: this is per-user; nobody else should be poking at it.
    std::fs::set_permissions(&socket_path, std::fs::Permissions::from_mode(0o600))
        .with_context(|| format!("chmod 0600 {}", socket_path.display()))?;

    info!(path = %socket_path.display(), "IPC listener up");

    let handle = tokio::spawn(async move {
        loop {
            match listener.accept().await {
                Ok((stream, _addr)) => {
                    let state = state.clone();
                    tokio::spawn(async move {
                        if let Err(e) = handle_conn(state, stream).await {
                            // EPIPE is the normal case for the queued-paste
                            // path: we reply `{"queued":true}` synchronously,
                            // then dispatch detached. If the trigger closed
                            // before we finished writing the reply, that's
                            // its 150 ms READ_TIMEOUT — not a bug. Demote so
                            // it doesn't pollute the WARN stream.
                            let msg = e.to_string();
                            if msg.contains("Broken pipe") || msg.contains("os error 32") {
                                debug!(error = %e, "IPC connection closed by client (likely trigger read timeout)");
                            } else {
                                warn!(error = %e, "IPC connection error");
                            }
                        }
                    });
                }
                Err(e) => {
                    error!(error = %e, "IPC accept failed");
                    // Tight loop is fine; if the listener itself is broken
                    // we'd rather see a flood in logs than silently die.
                    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                }
            }
        }
    });

    Ok(handle)
}

async fn handle_conn(state: Arc<SharedState>, mut stream: UnixStream) -> Result<()> {
    let started = Instant::now();

    // Length-prefixed read.
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf).await.context("read len")?;
    let len = u32::from_le_bytes(len_buf);
    if len == 0 || len > MAX_REQUEST_BYTES {
        write_response(
            &mut stream,
            &json!({ "ok": false, "reason": "bad-length", "fallback": "bash" }),
        )
        .await?;
        anyhow::bail!("bad request length: {len}");
    }
    let mut body = vec![0u8; len as usize];
    stream.read_exact(&mut body).await.context("read body")?;
    let req: Request = serde_json::from_slice(&body).context("parse request")?;

    let response = match req {
        Request::Paste { pane, .. } => handle_paste(&state, &pane, started).await,
        Request::Stage { image_path } => handle_stage(&state, &image_path).await,
        Request::StageText { bytes_b64, .. } => handle_stage_text(&state, &bytes_b64).await,
        Request::Ping => json!({ "ok": true, "pong": true }),
    };
    write_response(&mut stream, &response).await?;
    Ok(())
}

async fn write_response(stream: &mut UnixStream, value: &Value) -> Result<()> {
    let body = serde_json::to_vec(value)?;
    let len = u32::try_from(body.len()).context("response too large")?;
    stream.write_all(&len.to_le_bytes()).await?;
    stream.write_all(&body).await?;
    stream.flush().await?;
    Ok(())
}

async fn handle_paste(state: &Arc<SharedState>, pane: &str, started: Instant) -> Value {
    info!(pane, "paste: request received");

    // ─── Eager fresh-screenshot pickup ────────────────────────────────
    // GNOME's screenshot tool writes the PNG via a tempfile, atomically
    // renames it into place, but then *keeps the file descriptor open*
    // for ~3–5 seconds while it renders its in-shell "Screenshot saved"
    // notification. Inotify's `CLOSE_WRITE` doesn't fire until that fd
    // closes — so a paste right after PrtScr sees the old staged_text
    // and dispatches that instead of the brand-new screenshot.
    //
    // Defence in depth: on every paste, stat the screenshots dir for the
    // freshest PNG/JPG. If its mtime is newer than what the daemon has
    // staged, read it now and stage it. The dir scan is ~1ms on a
    // typical screenshots folder; the file read is ~5-20ms for a 500KB
    // PNG. Cost is acceptable; correctness is critical.
    eager_screenshot_pickup(state, pane).await;
    eager_live_image_pickup(state, pane).await;

    // ─── Intent: text or image (most-recent staged wins) ──────────────
    // The staged-selection slot is single-valued (set_staged_image
    // replaces text and vice versa), so whatever's in the slot is the
    // most-recent staged action. If the slot is empty / stale, fall back
    // to scraping the live X11 clipboard for fresh text. Otherwise punt
    // to bash. Dispatch itself still has a short duplicate-trigger guard
    // so one physical paste gesture cannot insert the same payload twice.
    let staged = resolve_paste_intent(state, pane).await;

    match staged {
        Some(StagedSelection::Text(text)) => {
            if !claim_paste_slot(
                state,
                paste_signature(pane, &text.bytes),
                PASTE_DEDUP_WINDOW_MS,
            ) {
                return deduped_response(pane, started);
            }
            let bytes = text.bytes.len();
            if let Err(e) = paste::dispatch_text_paste(state.clone(), pane.to_string(), text).await
            {
                error!(error = ?e, pane, "text paste dispatch failed");
                return json!({ "ok": false, "reason": "dispatch-failed", "fallback": "bash" });
            }
            json!({
                "ok": true,
                "kind": "text",
                "bytes": bytes as u64,
                "latency_ms": started.elapsed().as_millis() as u64,
            })
        }
        Some(StagedSelection::Image(img)) if img.is_fresh() => {
            if !claim_paste_slot(
                state,
                paste_signature(pane, &img.bytes),
                PASTE_DEDUP_WINDOW_IMAGE_MS,
            ) {
                return deduped_response(pane, started);
            }
            if let Err(e) = paste::dispatch_image_paste(state.clone(), pane.to_string(), img).await
            {
                error!(error = ?e, pane, "image paste dispatch failed");
                return json!({ "ok": false, "reason": "dispatch-failed", "fallback": "bash" });
            }
            json!({
                "ok": true,
                "kind": "image",
                "latency_ms": started.elapsed().as_millis() as u64,
            })
        }
        Some(StagedSelection::Image(_)) => {
            warn!(pane, "paste: staged image too old; punting to bash");
            json!({ "ok": false, "reason": "stale-image", "fallback": "bash" })
        }
        None => {
            info!(
                pane,
                "paste: nothing staged and clipboard has no text; punting to bash"
            );
            json!({ "ok": false, "reason": "no-content", "fallback": "bash" })
        }
    }
}

/// Eager fresh-screenshot pickup. GNOME's screenshot tool keeps the PNG fd
/// open for ~3-5s while rendering its "saved" toast, so inotify's CLOSE_WRITE
/// lags; a paste right after PrtScr would otherwise dispatch the stale staged
/// selection. On every paste we stat the screenshots dir for the freshest
/// image and, if it's newer than what's staged (and no fresh Wayland text
/// overrides it), stage it now. Extracted verbatim from handle_paste.
async fn eager_screenshot_pickup(state: &Arc<SharedState>, pane: &str) {
    if should_scan_screenshots(state) {
        if let Some((fresh_path, fresh_mtime)) = state
            .config
            .screenshots_dir
            .as_ref()
            .and_then(|dir| newest_image_in(dir))
        {
            let need_pickup = match state.staged_snapshot().await {
                Some(StagedSelection::Image(img)) => fresh_mtime > img.captured_at,
                Some(StagedSelection::Text(txt)) => fresh_mtime > txt.captured_at,
                _ => fresh_mtime
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs() > now_unix_secs().saturating_sub(60))
                    .unwrap_or(false),
            };
            if need_pickup {
                if let Some(text) = read_wayland_text_if_present().await {
                    let s = StagedText {
                        bytes: Arc::new(text),
                        captured_at: std::time::SystemTime::now(),
                    };
                    info!(
                        pane,
                        path = %fresh_path.display(),
                        bytes = s.bytes.len(),
                        "paste: fresh Wayland text suppressed eager screenshot pickup"
                    );
                    state.set_staged_text(s).await;
                } else if let Ok(bytes) = tokio::fs::read(&fresh_path).await {
                    let len = bytes.len();
                    let new_img = StagedImage {
                        bytes: Arc::new(bytes),
                        mime: mime_for_path(&fresh_path),
                        path: fresh_path.clone(),
                        captured_at: fresh_mtime,
                    };
                    info!(
                        pane,
                        path = %fresh_path.display(),
                        bytes = len,
                        "paste: eagerly picked up fresh screenshot (inotify hadn't fired yet)"
                    );
                    state.set_staged_image(new_img).await;
                }
            }
        }
    }
}

/// Eager live-clipboard image pickup. Bridges "browser Copy Image"
/// (Firefox/Chrome write image bytes with no screenshot file), which the
/// inotify watcher and dir scan both miss. Probe the live clipboard; if it
/// advertises an image whose bytes differ from what's staged, stage it now.
/// Extracted verbatim from handle_paste.
async fn eager_live_image_pickup(state: &Arc<SharedState>, pane: &str) {
    // ─── Eager live-clipboard image pickup ────────────────────────────
    // Bridges "browser Copy Image" — Firefox/Chrome under Wayland write
    // image bytes to the clipboard with no file under
    // ~/Pictures/Screenshots/, so the inotify watcher and the
    // newest-file scan above both miss them. Probe the live clipboard
    // here; if it advertises an image whose bytes differ from whatever
    // we have staged (or nothing's staged), stage them now. From this
    // point the existing X11/Wayland ownership path serves Claude
    // exactly the same way it does for inotify-staged screenshots.
    //
    // Cost on the happy path (no image on clipboard): one wl-clipboard
    // MIME probe (~5 ms) + xclip TARGETS probe (~10 ms) = ~15 ms.
    // Cost when capturing a 1 MB browser image: probe + read +
    // magic-check ~40–80 ms. Throttled to the same hot-path window as
    // the screenshots-dir scan so paste spam doesn't fan out probes.
    if should_probe_live_image(state) {
        if let Some((bytes, mime)) = read_clipboard_image_if_present().await {
            let differs = match state.staged_snapshot().await {
                Some(StagedSelection::Image(img)) => img.bytes.as_slice() != bytes.as_slice(),
                _ => true,
            };
            if differs {
                let len = bytes.len();
                // Synthetic path so downstream logging / Aider delivery
                // have a stable name. Use the screenshots dir when one
                // is configured so the file root is predictable.
                let path = state
                    .config
                    .screenshots_dir
                    .clone()
                    .unwrap_or_else(|| std::path::PathBuf::from("/tmp"))
                    .join(match mime {
                        "image/jpeg" => "flashpaste-clip-live.jpg",
                        "image/webp" => "flashpaste-clip-live.webp",
                        _ => "flashpaste-clip-live.png",
                    });
                let img = StagedImage {
                    bytes: Arc::new(bytes),
                    mime,
                    path,
                    captured_at: std::time::SystemTime::now(),
                };
                info!(
                    pane,
                    bytes = len,
                    mime,
                    "paste: captured live clipboard image (browser Copy Image bridge)"
                );
                state.set_staged_image(img).await;
            }
        }
    }
}

/// Resolve the paste intent: which selection (text or image) to dispatch.
/// Starts from the staged slot, then applies the live-clipboard override
/// cascade — scrape X11 text when nothing fresh is staged; let Wayland/X11
/// text override a staged image once it's old enough; let live text override
/// stale staged text. Returns the selection to dispatch. Extracted verbatim
/// from handle_paste so the hot path reads as three named phases.
async fn resolve_paste_intent(state: &Arc<SharedState>, pane: &str) -> Option<StagedSelection> {
    let mut staged = state.staged_snapshot().await;
    if !matches!(&staged, Some(s) if s.is_fresh()) {
        // Always read the live clipboard — never gate this behind the probe
        // throttle. On Mutter the Wayland read is wedged, so this X11 (xclip)
        // read is the ONLY way to notice the user copied fresh text since the
        // daemon last staged. A throttled skip here pastes stale/empty content
        // instead (the "URDF copy pastes empty" bug). xclip is ~10ms and
        // dock-flash-free; double-paste is guarded separately by the
        // content-hash dedup, not by this throttle.
        let external_text = read_clipboard_text_if_present().await;
        if let Some(bytes) = external_text {
            let s = StagedText {
                bytes: Arc::new(bytes),
                captured_at: std::time::SystemTime::now(),
            };
            info!(
                pane,
                bytes = s.bytes.len(),
                "paste: scraped X11 text → staged"
            );
            state.set_staged_text(s.clone()).await;
            staged = Some(StagedSelection::Text(s));
        }
    } else if let Some(StagedSelection::Image(img)) = &staged {
        // External-text override: if some other app now owns the live X11
        // CLIPBOARD with text-only targets, the user has copied text since
        // the daemon staged its image. Honour the user's intent.
        //
        // BUT skip the probe entirely if the image is *very fresh* (just
        // staged via inotify in the last 3 seconds). On Mutter the daemon's
        // X11 re-claim is asynchronous — wakes up `x11.rs` via the stage
        // notifier and re-issues SetSelectionOwner — and can lag behind
        // the staging event by 100-300ms in the average case (more for
        // 4K screenshots that the compressor was working on). Without
        // this gate the probe sees the *old* text targets (because the
        // re-claim hasn't propagated yet), returns true, and overrides
        // the brand-new screenshot back to stale text. Symptom: "I took
        // a screenshot and pasted right after, but it pasted the URL I
        // had on the clipboard before."
        //
        // Heuristic: any explicit user text-copy after the screenshot
        // is going to take at least ~1 second of human action (move
        // mouse, drag-select, release). 3 s is generous; longer than
        // any X11 re-claim race but shorter than any deliberate workflow.
        let age = img
            .captured_at
            .elapsed()
            .unwrap_or(std::time::Duration::ZERO);
        let wayland_text = read_wayland_text_if_present().await;
        if let Some(bytes) = wayland_text {
            let s = StagedText {
                bytes: Arc::new(bytes),
                captured_at: std::time::SystemTime::now(),
            };
            info!(
                pane,
                bytes = s.bytes.len(),
                image_age_ms = age.as_millis() as u64,
                "paste: live Wayland text overrides staged image"
            );
            state.set_staged_text(s.clone()).await;
            staged = Some(StagedSelection::Text(s));
        } else if age > std::time::Duration::from_secs(3) {
            if let Some(bytes) = read_clipboard_text_if_present().await {
                let s = StagedText {
                    bytes: Arc::new(bytes),
                    captured_at: std::time::SystemTime::now(),
                };
                info!(
                    pane,
                    bytes = s.bytes.len(),
                    image_age_s = age.as_secs(),
                    "paste: external X11 text overrides staged image (image old enough that user text-copy probably came after it)"
                );
                state.set_staged_text(s.clone()).await;
                staged = Some(StagedSelection::Text(s));
            }
        } else {
            debug!(
                pane,
                image_age_ms = age.as_millis() as u64,
                "paste: skipping external-text override — staged image is fresh, X11 re-claim may still be propagating"
            );
        }
    } else if let Some(StagedSelection::Text(existing)) = &staged {
        // Stale-text override: the daemon's `staged_text` only refreshes
        // when something calls `flashpaste-trigger --stage-text` (i.e.
        // clipboard-set.sh via tmux's `@clip` pipe on mouse-highlight).
        // When the user copies via kitty's `copy_and_clear_or_interrupt`
        // (Ctrl+C with a live selection) or any non-tmux app — e.g. a
        // browser text-copy from the X/Twitter composer — the bytes go
        // straight to the live clipboard and the daemon is never told.
        // Result: every paste delivers the OLD `staged_text` even though
        // the live clipboard holds bytes the user just copied.
        //
        // Probe Wayland FIRST, then fall back to X11 — same policy the
        // staged-image branch above uses. On Mutter a native-Wayland
        // app's Ctrl+C lands authoritatively on the Wayland clipboard,
        // while the X11 mirror stays sticky on whatever last claimed it
        // (here: the xclip owner clipboard-set.sh installed for the
        // highlighted terminal selection). An X11-only probe reads that
        // stale mirror, sees `bytes == existing`, skips the override, and
        // pastes the previously-highlighted text instead of the fresh
        // copy — the exact symptom this branch is meant to prevent.
        let live_text = match read_wayland_text_if_present().await {
            Some(b) => Some(b),
            None => read_clipboard_text_if_present().await,
        };
        if let Some(bytes) = live_text {
            if bytes.as_slice() != existing.bytes.as_slice() {
                let s = StagedText {
                    bytes: Arc::new(bytes),
                    captured_at: std::time::SystemTime::now(),
                };
                info!(
                    pane,
                    live_bytes = s.bytes.len(),
                    staged_bytes = existing.bytes.len(),
                    "paste: live clipboard text differs from staged_text — using live (kitty Ctrl+C, browser copy, or external app updated the clipboard)"
                );
                state.set_staged_text(s.clone()).await;
                staged = Some(StagedSelection::Text(s));
            }
        }
    }

    staged
}

fn deduped_response(pane: &str, started: Instant) -> Value {
    info!(pane, "paste: duplicate trigger absorbed");
    json!({
        "ok": true,
        "deduped": true,
        "latency_ms": started.elapsed().as_millis() as u64,
    })
}

/// Stable hash of `(pane, content)` — the dedup key. Two triggers for the
/// same paste carry identical pane + bytes, so they collide here; a
/// different pane or different bytes does not.
fn paste_signature(pane: &str, content: &[u8]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    pane.hash(&mut h);
    // Normalize trailing line endings before hashing. The dual-handler
    // delivers the *same* text via two paths whose trailing-newline handling
    // differs — one appends a `\n`, so the two fires render as "+N lines" and
    // "+N+1 lines" and, without this trim, hash to DIFFERENT signatures and
    // dodge the dedup. Trimming trailing `\n`/`\r` makes the near-identical
    // second fire collide on the same signature so it is absorbed.
    let trimmed = {
        let mut end = content.len();
        while end > 0 && matches!(content[end - 1], b'\n' | b'\r') {
            end -= 1;
        }
        &content[..end]
    };
    trimmed.hash(&mut h);
    h.finish()
}

/// Claim the dispatch slot for `(pane, content)`. Returns `false` (deduped)
/// when an identical signature arrived inside `PASTE_DEDUP_WINDOW_MS`.
/// The whole read-compare-store runs under the mutex so two simultaneous
/// identical triggers can't both claim.
fn claim_paste_slot(state: &SharedState, sig: u64, window_ms: u64) -> bool {
    let now = now_unix_ms();
    let mut last = state.last_paste.lock().unwrap_or_else(|e| e.into_inner());
    let (accept, next) = dedup_decide(*last, now, sig, window_ms);
    *last = next;
    accept
}

/// Pure dedup decision, split out of `claim_paste_slot` so the burst
/// behaviour is unit-testable without building a whole `SharedState`.
///
/// The window **slides**: an absorbed duplicate refreshes the timestamp just
/// like an accepted paste does. That is the fix for the observed double-image
/// paste. The old code anchored the window at the last *accepted* paste and
/// left the slot untouched when it absorbed one, so a burst of presses spread
/// wider than the window pasted twice. Measured on this box (journal,
/// 09:55:51.9 / :54.2 / :55.5, identical 236 KB screenshot into pane %5):
/// press 1 pasted, press 2 was absorbed at +2.31 s, press 3 landed at +3.58 s
/// — past the 2.5 s anchor — and pasted the *same* image a second time.
/// Sliding the window means the run collapses to one paste as long as no gap
/// inside it exceeds the window, which is exactly the "I pressed Ctrl+V a few
/// times because it felt slow" case. Pasting the same bytes into the same pane
/// on purpose still works; it just needs `PASTE_DEDUP_WINDOW_MS` of quiet.
fn dedup_decide(
    prev: Option<(u64, u64)>,
    now_ms: u64,
    sig: u64,
    window_ms: u64,
) -> (bool, Option<(u64, u64)>) {
    if let Some((last_ms, last_sig)) = prev {
        if last_sig == sig && is_duplicate_paste(now_ms, last_ms, window_ms) {
            return (false, Some((now_ms, sig)));
        }
    }
    (true, Some((now_ms, sig)))
}

fn is_duplicate_paste(now_ms: u64, last_ms: u64, window_ms: u64) -> bool {
    last_ms != 0 && now_ms.saturating_sub(last_ms) < window_ms
}

fn should_scan_screenshots(state: &SharedState) -> bool {
    throttle_ms(
        &state.last_screenshot_scan_ms,
        crate::tmux::HOT_PATH_PROBE_THROTTLE_MS,
    )
}

fn should_probe_live_image(state: &SharedState) -> bool {
    throttle_ms(
        &state.last_live_image_probe_ms,
        crate::tmux::HOT_PATH_PROBE_THROTTLE_MS,
    )
}

fn throttle_ms(slot: &std::sync::atomic::AtomicU64, min_interval_ms: u64) -> bool {
    let now = now_unix_ms();
    let last = slot.load(Ordering::Relaxed);
    if now.saturating_sub(last) < min_interval_ms {
        return false;
    }
    slot.compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed)
        .is_ok()
}

async fn handle_stage_text(state: &Arc<SharedState>, bytes_b64: &str) -> Value {
    let bytes = match decode_base64(bytes_b64) {
        Ok(b) => b,
        Err(e) => {
            return json!({
                "ok": false,
                "reason": "bad-base64",
                "detail": format!("{e:#}"),
            });
        }
    };
    let n = bytes.len();
    if bytes.is_empty() {
        // Never stage a zero-byte text selection. An empty selection carries no
        // clipboard payload; staging it would make the daemon's selection owners
        // advertise empty text/plain + UTF8_STRING targets, so the next paste
        // into any app delivers a zero-byte string. Every other text-staging
        // path already guards empty (read_clipboard_text_if_present,
        // read_wayland_text_if_present); this decoupled ingress now does too.
        // (`flashpaste-trigger --stage-text` short-circuits empty stdin before
        // it ever reaches us, so this guards any other caller of the op.)
        debug!("stage_text rejected: empty selection");
        return json!({ "ok": false, "reason": "empty-text", "bytes": 0 });
    }
    let staged = StagedText {
        bytes: Arc::new(bytes),
        captured_at: std::time::SystemTime::now(),
    };
    state.set_staged_text(staged).await;
    debug!(bytes = n, "stage_text accepted");
    json!({ "ok": true, "staged": "text", "bytes": n })
}

/// Read the text bytes currently on the X11 CLIPBOARD if (and only if)
/// the clipboard advertises a text-only target set (no image targets).
/// Used by `handle_paste` to scrape user-copied text into the daemon's
/// staged-text slot so subsequent pastes can dispatch from staged state
/// without depending on transient X11 selection ownership.
///
/// Returns `None` if:
///   * xclip TARGETS fails (no X server, no selection owner, etc.)
///   * any `image/*` target is advertised (the daemon's own image win)
///   * no text target is advertised
///   * xclip text read returns empty bytes
pub(crate) async fn read_clipboard_text_if_present() -> Option<Vec<u8>> {
    // First pass: TARGETS. Only proceed if it's text-only.
    let targets_out = tokio::process::Command::new("xclip")
        .args(["-selection", "clipboard", "-t", "TARGETS", "-o"])
        .output()
        .await
        .ok()?;
    if !targets_out.status.success() {
        return None;
    }
    let targets = String::from_utf8_lossy(&targets_out.stdout);
    let mut has_text_target: Option<&str> = None;
    let mut has_image = false;
    for line in targets.lines() {
        let t = line.trim();
        if t.starts_with("image/") {
            has_image = true;
            break;
        }
        if has_text_target.is_none() {
            if t == "UTF8_STRING" {
                has_text_target = Some("UTF8_STRING");
            } else if t.starts_with("text/plain") {
                has_text_target = Some(t.split(';').next().unwrap_or("text/plain"));
            } else if t == "STRING" {
                has_text_target = Some("STRING");
            }
        }
    }
    if has_image {
        return None;
    }
    let target = has_text_target?;
    // Read the actual bytes for the chosen target.
    let read_out = tokio::process::Command::new("xclip")
        .args(["-selection", "clipboard", "-t", target, "-o"])
        .output()
        .await
        .ok()?;
    if !read_out.status.success() || read_out.stdout.is_empty() {
        return None;
    }
    // Mutter's wedged bridge can return PNG/binary bytes on a text target.
    // Never stage an image as text — that's the raw-bytes-in-the-prompt bug.
    if !looks_like_text(&read_out.stdout) {
        warn!(
            bytes = read_out.stdout.len(),
            "paste: clipboard 'text' target returned non-text bytes (image leak) — ignoring"
        );
        return None;
    }
    Some(read_out.stdout)
}

/// Read text from the live Wayland clipboard only when Wayland clearly
/// advertises text and no image target. This is intentionally separate from
/// the X11 fallback: on Mutter, X11 can lag behind a fresh screenshot claim,
/// while Wayland is the authoritative signal that a browser text copy really
/// happened after the screenshot.
///
/// We request a specific plain-text MIME rather than `MimeType::Text` so
/// that wl_clipboard_rs cannot silently fall through to `text/html`. Sites
/// like Facebook offer both `text/plain` and `text/html` on their clipboard;
/// `MimeType::Text` may select the HTML variant, which contains raw
/// `<img src="blob:..."/>` markup that ends up verbatim in the chat prompt.
async fn read_wayland_text_if_present() -> Option<Vec<u8>> {
    let task = tokio::task::spawn_blocking(|| {
        use wl_clipboard_rs::paste::{get_contents, get_mime_types, ClipboardType, MimeType, Seat};

        let types = get_mime_types(ClipboardType::Regular, Seat::Unspecified).ok()?;
        if types.iter().any(|t| t.starts_with("image/")) {
            return None;
        }

        // Explicit preference order — plain text only, never text/html.
        // `MimeType::Text` would let the library pick freely and can select
        // text/html when a browser offers multiple text targets.
        const PREFER: &[&str] = &[
            "UTF8_STRING",
            "text/plain;charset=utf-8",
            "text/plain",
            "TEXT",
            "STRING",
        ];
        let mime = PREFER
            .iter()
            .find(|&&want| types.iter().any(|t| t == want))?;

        let (mut pipe, _mime) = get_contents(
            ClipboardType::Regular,
            Seat::Unspecified,
            MimeType::Specific(mime),
        )
        .ok()?;
        let mut bytes = Vec::new();
        pipe.read_to_end(&mut bytes).ok()?;
        // Same guard as the X11 reader: reject image bytes leaking onto a
        // Wayland text target so they're never staged + pasted as text.
        if bytes.is_empty() || !looks_like_text(&bytes) {
            None
        } else {
            Some(bytes)
        }
    });

    tokio::time::timeout(Duration::from_millis(150), task)
        .await
        .ok()?
        .ok()
        .flatten()
}

// Only referenced by the unit test below; gated so a release-clippy build
// (which doesn't compile tests) doesn't flag it as dead code.
#[cfg(test)]
fn is_text_target(target: &str) -> bool {
    matches!(target, "UTF8_STRING" | "STRING" | "TEXT") || target.starts_with("text/plain")
}

/// Reject clipboard bytes that are clearly NOT text before they get staged and
/// pasted as text.
///
/// On this GNOME-46/Mutter box the wedged X11<->Wayland bridge will hand back
/// raw image bytes on a *text* target after a screenshot (xclip's silent
/// fallback). Without this guard those bytes were staged as `StagedText` and
/// dumped verbatim into the prompt — a multi-KB wall of `\x89PNG…IEND` garbage.
/// Real clipboard text is valid UTF-8 with no NUL bytes; every image payload
/// fails one of those on its leading bytes, so this is both precise and cheap.
pub(crate) fn looks_like_text(bytes: &[u8]) -> bool {
    if bytes.is_empty() {
        return false;
    }
    // Fast path: known image magic numbers seen leaking onto text targets here.
    const IMAGE_MAGIC: &[&[u8]] = &[
        b"\x89PNG\r\n\x1a\n", // PNG
        b"\xff\xd8\xff",      // JPEG
        b"GIF87a",
        b"GIF89a",
        b"RIFF", // WebP / other RIFF container
    ];
    if IMAGE_MAGIC.iter().any(|m| bytes.starts_with(m)) {
        return false;
    }
    // Text never carries a NUL; binary almost always does in its first bytes.
    if bytes.contains(&0) {
        return false;
    }
    // Final gate: a real text selection decodes as UTF-8.
    std::str::from_utf8(bytes).is_ok()
}

/// Read raw image bytes off the live clipboard if (and only if) image MIME
/// is advertised and the returned bytes pass a magic-byte check. Wayland
/// first, X11 fallback. Returns `(bytes, mime)` on success.
///
/// Bridges the "browser Copy Image" case: Firefox/Chrome right-click →
/// "Copy Image" puts bytes on the clipboard with no file write, so the
/// daemon's inotify path on `~/Pictures/Screenshots/` never sees them.
/// Before this probe, the only fix was the `flashpaste-capture-clip`
/// shell shim called from `~/paste_image.sh` — but the kitty Ctrl+V
/// binding short-circuits to `flashpaste-trigger` whenever the user is
/// inside tmux, bypassing the shim entirely. Doing the probe here means
/// the daemon serves the bridge regardless of which path triggered the
/// paste.
///
/// MIME preference is PNG > JPEG > WebP. PNG is universally accepted by
/// Claude Code; JPEG is what most sites serve photographic content as;
/// WebP shows up on modern image-heavy sites. We pick the first one the
/// clipboard actually advertises and trust the magic check downstream
/// to catch xclip's silent text-fallback (xclip returns text bytes when
/// the requested MIME isn't really there, which would otherwise stage a
/// UTF-8 URL as if it were a PNG).
pub(crate) async fn read_clipboard_image_if_present() -> Option<(Vec<u8>, &'static str)> {
    const PREFERRED: &[&str] = &["image/png", "image/jpeg", "image/webp"];

    // ─── MIME probe: Wayland ────────────────────────────────────────
    let wl_types_task = tokio::task::spawn_blocking(|| {
        use wl_clipboard_rs::paste::{get_mime_types, ClipboardType, Seat};
        get_mime_types(ClipboardType::Regular, Seat::Unspecified)
            .ok()
            .map(|set| set.into_iter().collect::<Vec<String>>())
            .unwrap_or_default()
    });
    let wl_types = tokio::time::timeout(Duration::from_millis(150), wl_types_task)
        .await
        .ok()
        .and_then(|r| r.ok())
        .unwrap_or_default();

    // ─── MIME probe: X11 ────────────────────────────────────────────
    let x_types: Vec<String> = match tokio::time::timeout(
        Duration::from_millis(300),
        tokio::process::Command::new("xclip")
            .args(["-selection", "clipboard", "-t", "TARGETS", "-o"])
            .output(),
    )
    .await
    {
        Ok(Ok(out)) if out.status.success() => String::from_utf8_lossy(&out.stdout)
            .lines()
            .map(|s| s.trim().to_string())
            .collect(),
        _ => Vec::new(),
    };

    let has_image_wl = wl_types.iter().any(|t| t.starts_with("image/"));
    let has_image_x = x_types.iter().any(|t| t.starts_with("image/"));
    if !has_image_wl && !has_image_x {
        return None;
    }

    // Pick the best MIME advertised on either side.
    let mime: &'static str = if let Some(m) = PREFERRED
        .iter()
        .find(|want| wl_types.iter().any(|t| t == *want) || x_types.iter().any(|t| t == *want))
    {
        match *m {
            "image/png" => "image/png",
            "image/jpeg" => "image/jpeg",
            "image/webp" => "image/webp",
            _ => return None,
        }
    } else {
        return None;
    };

    // ─── Read bytes: Wayland first ──────────────────────────────────
    let bytes: Option<Vec<u8>> = if has_image_wl {
        let mime_owned = mime.to_string();
        let task = tokio::task::spawn_blocking(move || -> Option<Vec<u8>> {
            use wl_clipboard_rs::paste::{get_contents, ClipboardType, MimeType, Seat};
            let (mut pipe, _) = get_contents(
                ClipboardType::Regular,
                Seat::Unspecified,
                MimeType::Specific(&mime_owned),
            )
            .ok()?;
            let mut buf = Vec::new();
            pipe.read_to_end(&mut buf).ok()?;
            if buf.is_empty() {
                None
            } else {
                Some(buf)
            }
        });
        tokio::time::timeout(Duration::from_millis(500), task)
            .await
            .ok()
            .and_then(|r| r.ok())
            .flatten()
    } else {
        None
    };

    // ─── X11 fallback when Wayland returned empty ───────────────────
    let bytes = match bytes {
        Some(b) => Some(b),
        None if has_image_x => match tokio::time::timeout(
            Duration::from_millis(500),
            tokio::process::Command::new("xclip")
                .args(["-selection", "clipboard", "-t", mime, "-o"])
                .output(),
        )
        .await
        {
            Ok(Ok(out)) if out.status.success() && !out.stdout.is_empty() => Some(out.stdout),
            _ => None,
        },
        _ => None,
    };

    let bytes = bytes?;
    if !image_magic_ok(&bytes, mime) {
        return None;
    }
    Some((bytes, mime))
}

/// Validate that `bytes` actually starts with the magic prefix of the
/// declared MIME. Without this, xclip's silent text-fallback or a
/// truncated read could stage a text blob as a PNG. Identical in spirit
/// to the magic check in `bin/flashpaste-capture-clip`.
fn image_magic_ok(bytes: &[u8], mime: &str) -> bool {
    match mime {
        "image/png" => bytes.starts_with(b"\x89PNG\r\n\x1a\n"),
        "image/jpeg" => bytes.starts_with(b"\xff\xd8\xff"),
        "image/webp" => bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WEBP",
        _ => false,
    }
}

async fn handle_stage(state: &Arc<SharedState>, image_path: &str) -> Value {
    let path = std::path::PathBuf::from(image_path);
    let mime = match path
        .extension()
        .and_then(|s| s.to_str())
        .map(str::to_lowercase)
    {
        Some(ref ext) if ext == "png" => "image/png",
        Some(ref ext) if ext == "jpg" || ext == "jpeg" => "image/jpeg",
        _ => "image/png",
    };
    let bytes = match tokio::fs::read(&path).await {
        Ok(b) => b,
        Err(e) => {
            return json!({
                "ok": false,
                "reason": "read-failed",
                "detail": format!("{e:#}"),
            });
        }
    };
    let staged = StagedImage {
        bytes: Arc::new(bytes),
        mime,
        path,
        captured_at: std::time::SystemTime::now(),
    };
    state.set_staged_image(staged).await;
    json!({ "ok": true, "staged": true })
}

fn is_socket(p: &Path) -> bool {
    use std::os::unix::fs::FileTypeExt;
    match std::fs::symlink_metadata(p) {
        Ok(md) => md.file_type().is_socket(),
        Err(_) => false,
    }
}

/// Minimal standard-base64 decoder. Avoids pulling in the `base64` crate
/// for a single hot use. Accepts whitespace + ignores `=` padding length.
/// Find the most recently-modified PNG/JPEG file in `dir`. Returns the
/// path and its mtime. Used by `handle_paste` to eagerly pick up
/// screenshots that GNOME has written but whose `CLOSE_WRITE` event
/// hasn't fired yet (the screenshot tool keeps the fd open while
/// rendering its in-shell "saved" toast).
fn newest_image_in(dir: &std::path::Path) -> Option<(std::path::PathBuf, std::time::SystemTime)> {
    let entries = std::fs::read_dir(dir).ok()?;
    let mut best: Option<(std::time::SystemTime, std::path::PathBuf)> = None;
    for entry in entries.flatten() {
        let p = entry.path();
        let lower = p
            .extension()
            .and_then(|s| s.to_str())
            .map(str::to_lowercase);
        match lower.as_deref() {
            Some("png") | Some("jpg") | Some("jpeg") => {}
            _ => continue,
        }
        // Skip our own compressed siblings (`.fpc.<ext>`).
        if p.file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.contains(".fpc."))
        {
            continue;
        }
        let mtime = match entry.metadata().and_then(|m| m.modified()) {
            Ok(t) => t,
            Err(_) => continue,
        };
        match &best {
            Some((cur, _)) if *cur >= mtime => {}
            _ => best = Some((mtime, p)),
        }
    }
    best.map(|(t, p)| (p, t))
}

fn mime_for_path(p: &std::path::Path) -> &'static str {
    match p
        .extension()
        .and_then(|s| s.to_str())
        .map(str::to_lowercase)
        .as_deref()
    {
        Some("jpg") | Some("jpeg") => "image/jpeg",
        _ => "image/png",
    }
}

fn now_unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn decode_base64(input: &str) -> anyhow::Result<Vec<u8>> {
    let mut out = Vec::with_capacity(input.len() * 3 / 4);
    let mut buf: u32 = 0;
    let mut bits: u8 = 0;
    for ch in input.chars() {
        let val: u32 = match ch {
            'A'..='Z' => (ch as u32) - ('A' as u32),
            'a'..='z' => (ch as u32) - ('a' as u32) + 26,
            '0'..='9' => (ch as u32) - ('0' as u32) + 52,
            '+' => 62,
            '/' => 63,
            '=' | '\r' | '\n' | ' ' | '\t' => continue,
            _ => anyhow::bail!("invalid base64 char: {ch:?}"),
        };
        buf = (buf << 6) | val;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buf >> bits) as u8);
            buf &= (1u32 << bits) - 1;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_basic_base64() {
        assert_eq!(decode_base64("aGVsbG8=").unwrap(), b"hello");
        assert_eq!(decode_base64("aGVsbG8gd29ybGQ=").unwrap(), b"hello world");
        assert_eq!(decode_base64("").unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn looks_like_text_accepts_real_text() {
        assert!(looks_like_text(b"hello world"));
        assert!(looks_like_text(
            "https://lifted.sk/hu/products/gym".as_bytes()
        ));
        assert!(looks_like_text("ÁÉÍ Slovak ďáčik · €4.40".as_bytes()));
    }

    #[test]
    fn looks_like_text_rejects_image_and_binary() {
        // The exact regression: a PNG served on a "text" target.
        assert!(!looks_like_text(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"));
        assert!(!looks_like_text(b"\xff\xd8\xff\xe0JFIF")); // JPEG
        assert!(!looks_like_text(b"GIF89a\x01\x00")); // GIF
        assert!(!looks_like_text(b"RIFF\x00\x00WEBP")); // WebP
        assert!(!looks_like_text(b"text with a \x00 nul")); // NUL byte
        assert!(!looks_like_text(&[0xff, 0xfe, 0xfd])); // invalid UTF-8
        assert!(!looks_like_text(b"")); // empty is not pasteable text
    }

    #[test]
    fn ignores_whitespace_and_padding() {
        assert_eq!(decode_base64("aGVs\nbG8=").unwrap(), b"hello");
        assert_eq!(decode_base64("aGVsbG8 ").unwrap(), b"hello");
    }

    #[test]
    fn duplicate_paste_window_absorbs_only_recent_repeats() {
        // last_ms == 0 means "never pasted" — not a duplicate.
        assert!(!is_duplicate_paste(1_000, 0, PASTE_DEDUP_WINDOW_MS));
        // A repeat inside the window is absorbed — including the measured
        // 1.84 s kitty-router-vs-tmux-bind spread that the old 1 s window let
        // through.
        assert!(is_duplicate_paste(1_900, 1_000, PASTE_DEDUP_WINDOW_MS));
        assert!(is_duplicate_paste(2_800, 1_000, PASTE_DEDUP_WINDOW_MS)); // 1.8s gap
                                                                          // A repeat at exactly the window edge or beyond is a fresh paste.
        assert!(!is_duplicate_paste(3_500, 1_000, PASTE_DEDUP_WINDOW_MS));
        assert!(!is_duplicate_paste(3_501, 1_000, PASTE_DEDUP_WINDOW_MS));
    }

    /// Replay a run of triggers through the dedup slot, returning how many
    /// were accepted (i.e. how many pastes the user actually sees).
    ///
    /// Offsets are relative to a nonzero epoch base: `is_duplicate_paste`
    /// treats `last_ms == 0` as the "never pasted" sentinel, and real
    /// timestamps are unix-epoch ms, so a run must not start at 0.
    fn replay_window(sig: u64, offsets_ms: &[u64], window_ms: u64) -> usize {
        const BASE_MS: u64 = 1_774_000_000_000;
        let mut slot: Option<(u64, u64)> = None;
        let mut accepted = 0;
        for &t in offsets_ms {
            let (ok, next) = dedup_decide(slot, BASE_MS + t, sig, window_ms);
            slot = next;
            if ok {
                accepted += 1;
            }
        }
        accepted
    }

    fn replay(sig: u64, offsets_ms: &[u64]) -> usize {
        replay_window(sig, offsets_ms, PASTE_DEDUP_WINDOW_MS)
    }

    #[test]
    fn burst_of_identical_triggers_collapses_to_one_paste() {
        // The measured double-image-paste, replayed from the daemon journal:
        // three triggers for the SAME 236 KB screenshot into pane %5 at
        // 09:55:51.933 / :54.239 / :55.510. Press 2 lands at +2306 ms (inside
        // the window, absorbed); press 3 at +3577 ms is past the 2500 ms
        // window *measured from press 1* — which is how the old anchored
        // window let it through and pasted the image twice. With a sliding
        // window press 3 is only 1271 ms after press 2, so it is absorbed too.
        assert_eq!(replay(0xABCD, &[0, 2306, 3577]), 1);
    }

    #[test]
    fn long_impatient_burst_still_pastes_once() {
        // Ctrl+V held/mashed for ~4.5 s: every gap is under the window, so
        // the whole run is one paste no matter how long it goes on.
        assert_eq!(replay(0xABCD, &[0, 800, 1600, 2400, 3200, 4000, 4500]), 1);
    }

    #[test]
    fn deliberate_repaste_after_quiet_is_allowed() {
        // The window sliding must not make the same content un-pasteable
        // forever: once the user stops pressing for a full window, an
        // intentional repeat of the same bytes goes through.
        assert_eq!(replay(0xABCD, &[0, 2306, 3577, 6100]), 2);
    }

    #[test]
    fn image_window_absorbs_a_human_cadence_re_press() {
        // The text window alone is not enough for images. Replaying the burst
        // against the live daemon put press 2 at +2540 ms — 40 ms past the
        // 2500 ms text window — so the same screenshot was attached twice even
        // with the window sliding. The image window must absorb that, and a
        // slower mash (~3 s between presses) too.
        assert_eq!(
            replay_window(0xABCD, &[0, 2540, 4400], PASTE_DEDUP_WINDOW_IMAGE_MS),
            1
        );
        assert_eq!(
            replay_window(0xABCD, &[0, 3000, 6000, 9000], PASTE_DEDUP_WINDOW_IMAGE_MS),
            1
        );
        // The same run under the text window pastes TWICE — press 2 escapes at
        // +2540 ms and press 3 is then absorbed relative to it. That second
        // paste is the user-visible double, and it is exactly why images get
        // their own, wider window.
        assert_eq!(replay(0xABCD, &[0, 2540, 4400]), 2);
    }

    #[test]
    fn image_repaste_still_possible_after_quiet() {
        // Widening must not make a screenshot un-re-pasteable, just delayed.
        assert_eq!(
            replay_window(0xABCD, &[0, 2540, 8000], PASTE_DEDUP_WINDOW_IMAGE_MS),
            2
        );
    }

    #[test]
    fn distinct_content_is_never_deduped() {
        // A different signature always claims, even back-to-back — dedup is
        // keyed on content, so a real second paste is never swallowed.
        let (ok1, slot) = dedup_decide(None, 1_000, 0xAAAA, PASTE_DEDUP_WINDOW_MS);
        let (ok2, _) = dedup_decide(slot, 1_010, 0xBBBB, PASTE_DEDUP_WINDOW_MS);
        assert!(ok1 && ok2);
    }

    #[test]
    fn paste_signature_distinguishes_pane_and_content() {
        let a = paste_signature("%4", b"hello");
        // Same pane + same bytes => same signature (the dual-handler case).
        assert_eq!(a, paste_signature("%4", b"hello"));
        // Different pane => different signature (don't dedup across panes).
        assert_ne!(a, paste_signature("%5", b"hello"));
        // Different bytes => different signature (don't dedup distinct text).
        assert_ne!(a, paste_signature("%4", b"world"));
        // Trailing-newline variants collide: the dual-handler delivers the
        // same text with/without a trailing \n ("+N" vs "+N+1 lines"); both
        // must dedup to the same signature.
        assert_eq!(a, paste_signature("%4", b"hello\n"));
        assert_eq!(a, paste_signature("%4", b"hello\r\n"));
        assert_eq!(a, paste_signature("%4", b"hello\n\n"));
        // But an interior newline is real content, not trailing — keep distinct.
        assert_ne!(a, paste_signature("%4", b"hel\nlo"));
    }

    #[test]
    fn text_target_detection_matches_clipboard_aliases() {
        assert!(is_text_target("text/plain;charset=utf-8"));
        assert!(is_text_target("text/plain"));
        assert!(is_text_target("UTF8_STRING"));
        assert!(!is_text_target("text/html"));
        assert!(!is_text_target("image/png"));
    }
}
