//! flashpaste-trigger — the 1-byte hot path.
//!
//! Tmux's `bind -n C-v` calls this binary with the target `pane_id`. It tries
//! to talk to the long-lived `flashpasted` daemon over its unix socket at
//! `$XDG_RUNTIME_DIR/flashpaste.sock`. If the daemon is up, the daemon does
//! all the work (clipboard ownership, kitty IPC, tmux unbind/rebind dance,
//! recursion guard) and replies in ~10ms.
//!
//! If the socket doesn't exist OR the connect/handshake fails within ~5ms,
//! we `exec` (replace ourselves with) the bash dispatcher found via PATH or
//! `FLASHPASTE_BASH_FALLBACK`. That way the trigger is always safe to wire
//! into tmux even when the daemon is dead or being restarted.
//!
//! Hard constraints from the spec:
//!   * Minimal deps — clap, serde_json, nix, anyhow. No tokio. No x11rb.
//!     No wl-clipboard-rs. No notify/inotify.
//!   * Target binary size <500 KB stripped.
//!
//! Wire protocol (matches `flashpasted`'s `src/ipc.rs`):
//!   request:  4-byte LE u32 length || JSON bytes
//!   response: 4-byte LE u32 length || JSON bytes
//!
//! Request body:
//!   {"op":"paste","pane":"%4","ts":"2026-05-19T12:34:56.789Z"}
//!
//! Response body (on success):
//!   {"ok":true,"latency_ms":7}
//!   {"ok":true,"deduped":true}
//!
//! Response body (daemon punts back to bash):
//!   {"ok":false,"reason":"no-image","fallback":"bash"}

use std::ffi::OsString;
use std::io::{Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::UnixStream;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use clap::Parser;
use nix::unistd::Uid;
use serde_json::{json, Value};

/// Where the daemon listens.
fn socket_path() -> PathBuf {
    if let Ok(dir) = std::env::var("XDG_RUNTIME_DIR") {
        if !dir.is_empty() {
            return PathBuf::from(dir).join("flashpaste.sock");
        }
    }
    let uid = Uid::current().as_raw();
    let candidate = PathBuf::from(format!("/run/user/{uid}"));
    if candidate.is_dir() {
        return candidate.join("flashpaste.sock");
    }
    PathBuf::from("/tmp").join("flashpaste.sock")
}

/// Total budget for "is the daemon there and willing to take this paste?"
/// Anything slower and we'd rather fall through to bash than hang the user.
const CONNECT_TIMEOUT: Duration = Duration::from_millis(5);
const WRITE_TIMEOUT: Duration = Duration::from_millis(10);
/// How long we wait for the daemon's reply.
///
/// This used to be 150 ms, which was BELOW the daemon's real image-paste
/// latency and caused a user-visible double paste. The daemon replies only
/// after `dispatch_image_paste` finishes (X11 re-claim + `tmux if-shell` +
/// `select-pane` + `send-keys`), so the reply carries the full dispatch cost.
/// Measured on this box over one day (55 image pastes, daemon journal
/// `request received` → `PASTED image`): median 37 ms, p90 242 ms, max 341 ms
/// — 20% of image pastes were slower than 150 ms. Every one of those timed
/// out here, was reported as a daemon *error*, and exec'd the bash dispatcher,
/// which pasted the SAME screenshot a second time ~330 ms after the daemon's
/// paste. The daemon's `(pane, content)` dedup cannot catch that: the second
/// paste never goes through the socket.
///
/// 2 s is ~6x the observed worst case. Waiting is free: tmux invokes the
/// trigger through `run-shell -b`, so nothing user-facing blocks on it.
const READ_TIMEOUT: Duration = Duration::from_millis(2000);
/// Timeout for the liveness `ping` we send after a read timeout, to tell a
/// slow-but-working daemon apart from a wedged one. Ping is answered from a
/// fresh connection task, so it does not queue behind our own paste.
const PING_TIMEOUT: Duration = Duration::from_millis(300);

#[derive(Debug, Parser)]
#[command(
    name = "flashpaste-trigger",
    about = "Trigger a paste via the flashpasted daemon (with bash fallback)."
)]
struct Args {
    /// Target tmux pane id (e.g. `%4`). Required for the paste op; unused
    /// for --stage-text.
    pane: Option<String>,
    /// Op to send. Defaults to `paste`.
    #[arg(long, default_value = "paste")]
    op: String,
    /// Force the bash fallback path (skip the daemon entirely). For debugging.
    #[arg(long)]
    force_fallback: bool,
    /// Read stdin and stage as a TEXT selection in the daemon (v1.19+).
    /// This is the path `clipboard-set.sh` uses to avoid forking wl-copy
    /// (which surfaces as a phantom dock entry). On success, exits 0; on
    /// any failure (daemon down, write error), exits non-zero so the
    /// caller can fall back to wl-copy or another backend.
    #[arg(long)]
    stage_text: bool,
}

fn main() -> ! {
    let args = Args::parse();

    if args.stage_text {
        // Decoupled path — no pane needed, no bash fallback. The caller
        // (clipboard-set.sh) handles its own fallback if we exit non-zero.
        trigger_log("stage-text", "-", "start", "reading stdin");
        let code = stage_text_main();
        trigger_log("stage-text", "-", "exit", &format!("code={code}"));
        std::process::exit(code);
    }

    let pane = match &args.pane {
        Some(p) => p.clone(),
        None => {
            eprintln!("flashpaste-trigger: missing pane argument");
            std::process::exit(2);
        }
    };

    let trigger_source =
        std::env::var("TMUX_PASTE_TRIGGER").unwrap_or_else(|_| "unset".to_string());
    trigger_log(
        "paste",
        &pane,
        "start",
        &format!("trigger={trigger_source}"),
    );

    if args.force_fallback {
        trigger_log("paste", &pane, "exec-bash", "force-fallback");
        exec_bash_fallback(&pane);
    }

    let paste_args = PasteArgs {
        pane: pane.clone(),
        op: args.op,
    };
    match try_daemon(&paste_args) {
        Ok(DaemonOutcome::Handled) => {
            trigger_log("paste", &pane, "handled", "daemon");
            std::process::exit(0);
        }
        Ok(DaemonOutcome::AcceptedNoReply) => {
            // Slow daemon, still alive: it owns the paste. Exiting 0 also
            // stops tmux's `trigger || bash-dispatch` shell fallback from
            // firing the second paste behind our back.
            trigger_log("paste", &pane, "handled", "daemon-slow-no-reply");
            std::process::exit(0);
        }
        Ok(DaemonOutcome::FallbackRequested) => {
            trigger_log("paste", &pane, "exec-bash", "daemon-declined");
            exec_bash_fallback(&pane);
        }
        Err(e) => {
            trigger_log("paste", &pane, "exec-bash", &format!("daemon-error: {e}"));
            exec_bash_fallback(&pane);
        }
    }
}

/// Per-invocation log written to `$FLASHPASTE_TRIGGER_LOG` or
/// `~/.local/state/flashpaste-trigger.log`. The whole point of this log
/// is debugging "right-click Paste doesn't paste but Ctrl+V does":
/// every invocation appends one line so you can see when each handler
/// fires, what the trigger source was, and which path the daemon chose.
///
/// Suppress with `FLASHPASTE_QUIET=1`.
fn trigger_log(op: &str, pane: &str, phase: &str, detail: &str) {
    if std::env::var_os("FLASHPASTE_QUIET").is_some() {
        return;
    }
    if std::env::var_os("FLASHPASTE_TRIGGER_LOG").is_none()
        && std::env::var_os("FLASHPASTE_TRIGGER_DEBUG").is_none()
    {
        return;
    }
    let path = log_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    else {
        return;
    };
    let ts = iso8601_utc_now();
    let pid = std::process::id();
    let _ = writeln!(
        f,
        "{ts} pid={pid} op={op} pane={pane} phase={phase} :: {detail}"
    );
}

fn log_path() -> PathBuf {
    if let Ok(p) = std::env::var("FLASHPASTE_TRIGGER_LOG") {
        if !p.is_empty() {
            return PathBuf::from(p);
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home)
            .join(".local")
            .join("state")
            .join("flashpaste-trigger.log");
    }
    PathBuf::from("/tmp/flashpaste-trigger.log")
}

struct PasteArgs {
    pane: String,
    op: String,
}

enum DaemonOutcome {
    /// Daemon handled the paste end-to-end.
    Handled,
    /// Daemon explicitly told us to fall back (e.g. no staged image, text paste).
    FallbackRequested,
    /// The request was written to a daemon that is still alive, but the reply
    /// did not arrive inside `READ_TIMEOUT`. The daemon OWNS this paste — it
    /// has the request and its dispatch is what we are waiting on. Running the
    /// bash dispatcher here would paste the same content a second time, which
    /// is exactly the double-paste bug (the daemon's `(pane, content)` dedup
    /// is blind to a paste that never crosses the socket). So: no fallback.
    AcceptedNoReply,
}

fn try_daemon(args: &PasteArgs) -> Result<DaemonOutcome> {
    let path = socket_path();
    if !path.exists() {
        anyhow::bail!("no daemon socket at {}", path.display());
    }

    // connect_timeout doesn't exist for UnixStream in std, but the socket is
    // local and SOCK_STREAM connect on a listening unix socket completes in
    // microseconds. If it stalls, the read/write timeouts below catch it.
    let mut stream =
        UnixStream::connect(&path).with_context(|| format!("connect {}", path.display()))?;
    stream.set_write_timeout(Some(WRITE_TIMEOUT)).ok();
    stream.set_read_timeout(Some(READ_TIMEOUT)).ok();
    // Don't sit on Nagle; payloads are tiny.
    // (UnixStream uses AF_UNIX so TCP_NODELAY doesn't apply, but we don't
    // need anything: the kernel flushes on close anyway.)

    let _ = CONNECT_TIMEOUT; // budget hint for readers; std's API doesn't expose it.

    let req = json!({
        "op": args.op,
        "pane": args.pane,
        "ts": iso8601_utc_now(),
    });
    let body = serde_json::to_vec(&req)?;
    let len = u32::try_from(body.len()).context("request too large")?;
    stream.write_all(&len.to_le_bytes())?;
    stream.write_all(&body)?;
    // Half-close write side so the daemon can detect EOF if it wants.
    // (Optional; the daemon also honors the length prefix.)

    // From here on the daemon HAS our request. A read failure must not be
    // treated as "nothing happened": if the daemon is merely slow, it is
    // already pasting, and a bash fallback would double it.
    let mut len_buf = [0u8; 4];
    if let Err(e) = stream.read_exact(&mut len_buf) {
        if is_timeout(&e) && daemon_is_alive() {
            return Ok(DaemonOutcome::AcceptedNoReply);
        }
        return Err(anyhow::Error::new(e).context("read response length"));
    }
    let resp_len = u32::from_le_bytes(len_buf) as usize;
    // Bound the response so a wedged daemon can't make us allocate gigabytes.
    if resp_len > 64 * 1024 {
        anyhow::bail!("daemon response too large: {resp_len} bytes");
    }
    let mut resp_buf = vec![0u8; resp_len];
    if let Err(e) = stream.read_exact(&mut resp_buf) {
        if is_timeout(&e) && daemon_is_alive() {
            return Ok(DaemonOutcome::AcceptedNoReply);
        }
        return Err(anyhow::Error::new(e).context("read response body"));
    }
    let resp: Value = serde_json::from_slice(&resp_buf)?;

    let ok = resp.get("ok").and_then(Value::as_bool).unwrap_or(false);
    if ok {
        return Ok(DaemonOutcome::Handled);
    }

    // Daemon politely declined — fall through to bash.
    Ok(DaemonOutcome::FallbackRequested)
}

/// True when `e` is a socket read/write timeout. `SO_RCVTIMEO` surfaces as
/// `EAGAIN`/`EWOULDBLOCK` on Linux; other platforms use `TimedOut`.
fn is_timeout(e: &std::io::Error) -> bool {
    matches!(
        e.kind(),
        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
    )
}

/// Liveness probe: is the daemon still answering, or is it wedged?
///
/// Called only after our paste request timed out, to decide between "slow
/// daemon, it owns the paste, do nothing" and "dead daemon, bash must paste".
/// The daemon spawns a task per connection, so this ping does not queue
/// behind the slow paste we are still waiting on.
fn daemon_is_alive() -> bool {
    let path = socket_path();
    let Ok(mut stream) = UnixStream::connect(&path) else {
        return false;
    };
    stream.set_write_timeout(Some(PING_TIMEOUT)).ok();
    stream.set_read_timeout(Some(PING_TIMEOUT)).ok();

    let Ok(body) = serde_json::to_vec(&json!({ "op": "ping" })) else {
        return false;
    };
    let Ok(len) = u32::try_from(body.len()) else {
        return false;
    };
    if stream.write_all(&len.to_le_bytes()).is_err() || stream.write_all(&body).is_err() {
        return false;
    }
    let mut len_buf = [0u8; 4];
    if stream.read_exact(&mut len_buf).is_err() {
        return false;
    }
    let resp_len = u32::from_le_bytes(len_buf) as usize;
    if resp_len > 64 * 1024 {
        return false;
    }
    let mut resp_buf = vec![0u8; resp_len];
    if stream.read_exact(&mut resp_buf).is_err() {
        return false;
    }
    serde_json::from_slice::<Value>(&resp_buf)
        .ok()
        .and_then(|v| v.get("ok").and_then(Value::as_bool))
        .unwrap_or(false)
}

/// v1.19+: stage stdin as a text selection in the daemon. Returns the
/// process exit code: 0 on success, non-zero so the caller (typically
/// `clipboard-set.sh`) can fall back to a real wl-copy / xclip backend.
///
/// Cap stdin at 6 MB to match the daemon's MAX_REQUEST_BYTES budget.
fn stage_text_main() -> i32 {
    const MAX_STDIN_BYTES: usize = 6 * 1024 * 1024;

    let mut buf = Vec::new();
    let mut stdin = std::io::stdin().lock();
    if let Err(e) = read_to_vec_bounded(&mut stdin, &mut buf, MAX_STDIN_BYTES) {
        eprintln!("flashpaste-trigger --stage-text: read stdin: {e}");
        return 11;
    }
    if buf.is_empty() {
        // Nothing to stage; behave like success so the caller's fall-through
        // logic doesn't double-publish an empty clipboard.
        return 0;
    }

    let path = socket_path();
    if !path.exists() {
        return 12;
    }

    let stream_res = UnixStream::connect(&path);
    let mut stream = match stream_res {
        Ok(s) => s,
        Err(_) => return 13,
    };
    stream
        .set_write_timeout(Some(Duration::from_millis(200)))
        .ok();
    stream
        .set_read_timeout(Some(Duration::from_millis(200)))
        .ok();

    let b64 = base64_encode(&buf);
    let req = json!({
        "op": "stage_text",
        "bytes_b64": b64,
        "from": std::env::var("FLASHPASTE_STAGE_FROM").ok(),
    });
    let body = match serde_json::to_vec(&req) {
        Ok(b) => b,
        Err(_) => return 14,
    };
    let len = match u32::try_from(body.len()) {
        Ok(n) => n,
        Err(_) => return 15,
    };
    if stream.write_all(&len.to_le_bytes()).is_err() || stream.write_all(&body).is_err() {
        return 16;
    }
    let mut len_buf = [0u8; 4];
    if stream.read_exact(&mut len_buf).is_err() {
        return 17;
    }
    let resp_len = u32::from_le_bytes(len_buf) as usize;
    if resp_len > 64 * 1024 {
        return 18;
    }
    let mut resp_buf = vec![0u8; resp_len];
    if stream.read_exact(&mut resp_buf).is_err() {
        return 19;
    }
    let resp: Value = match serde_json::from_slice(&resp_buf) {
        Ok(v) => v,
        Err(_) => return 20,
    };
    let ok = resp.get("ok").and_then(Value::as_bool).unwrap_or(false);
    if ok {
        0
    } else {
        21
    }
}

/// Read up to `max` bytes from `r` into `out`. Returns an error if the
/// stream contains more than `max` bytes.
fn read_to_vec_bounded<R: Read>(r: &mut R, out: &mut Vec<u8>, max: usize) -> std::io::Result<()> {
    let mut tmp = [0u8; 8192];
    loop {
        let n = r.read(&mut tmp)?;
        if n == 0 {
            return Ok(());
        }
        if out.len() + n > max {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("stdin exceeds {max} bytes"),
            ));
        }
        out.extend_from_slice(&tmp[..n]);
    }
}

/// Standard base64 encoder (no URL-safe variant). Avoids the `base64`
/// crate so trigger stays minimal.
fn base64_encode(data: &[u8]) -> String {
    const CHARS: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    let mut i = 0;
    while i + 3 <= data.len() {
        let n = ((data[i] as u32) << 16) | ((data[i + 1] as u32) << 8) | (data[i + 2] as u32);
        out.push(CHARS[((n >> 18) & 0x3f) as usize] as char);
        out.push(CHARS[((n >> 12) & 0x3f) as usize] as char);
        out.push(CHARS[((n >> 6) & 0x3f) as usize] as char);
        out.push(CHARS[(n & 0x3f) as usize] as char);
        i += 3;
    }
    let rem = data.len() - i;
    if rem == 1 {
        let n = (data[i] as u32) << 16;
        out.push(CHARS[((n >> 18) & 0x3f) as usize] as char);
        out.push(CHARS[((n >> 12) & 0x3f) as usize] as char);
        out.push('=');
        out.push('=');
    } else if rem == 2 {
        let n = ((data[i] as u32) << 16) | ((data[i + 1] as u32) << 8);
        out.push(CHARS[((n >> 18) & 0x3f) as usize] as char);
        out.push(CHARS[((n >> 12) & 0x3f) as usize] as char);
        out.push(CHARS[((n >> 6) & 0x3f) as usize] as char);
        out.push('=');
    }
    out
}

/// Replace this process with the bash dispatcher. Zero overhead — the
/// trigger binary's pid simply becomes the bash script.
fn exec_bash_fallback(pane: &str) -> ! {
    let Some(fallback) = bash_fallback_path() else {
        eprintln!("flashpaste-trigger: tmux-paste-dispatch fallback not found");
        std::process::exit(127);
    };
    let mut cmd = Command::new(&fallback);
    cmd.arg(pane);
    // Propagate the trigger source so the bash script's logs distinguish
    // "real ctrl-v" from "daemon punted".
    let trigger_value: OsString = std::env::var_os("TMUX_PASTE_TRIGGER")
        .unwrap_or_else(|| OsString::from("flashpaste-trigger-fallback"));
    cmd.env("TMUX_PASTE_TRIGGER", trigger_value);

    // `exec` only returns on error.
    let err = cmd.exec();
    eprintln!(
        "flashpaste-trigger: exec {} failed: {}",
        fallback.display(),
        err
    );
    std::process::exit(127);
}

fn bash_fallback_path() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("FLASHPASTE_BASH_FALLBACK") {
        if !path.is_empty() {
            let candidate = PathBuf::from(path);
            if is_executable_file(&candidate) {
                return Some(candidate);
            }
        }
    }
    find_on_path(&["tmux-paste-dispatch", "tmux-paste-dispatch.sh"]).or_else(|| {
        let mut candidates = Vec::new();
        if let Some(home) = std::env::var_os("HOME") {
            let local_bin = PathBuf::from(home).join(".local").join("bin");
            candidates.push(local_bin.join("tmux-paste-dispatch"));
            candidates.push(local_bin.join("tmux-paste-dispatch.sh"));
        }
        candidates.push(PathBuf::from("/usr/bin/tmux-paste-dispatch"));
        candidates
            .into_iter()
            .find(|candidate| is_executable_file(candidate))
    })
}

fn find_on_path(names: &[&str]) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        for name in names {
            let candidate = dir.join(name);
            if is_executable_file(&candidate) {
                return Some(candidate);
            }
        }
    }
    None
}

fn is_executable_file(path: &Path) -> bool {
    std::fs::metadata(path)
        .map(|meta| meta.is_file() && meta.permissions().mode() & 0o111 != 0)
        .unwrap_or(false)
}

/// Minimal ISO-8601 UTC formatter — keeps us off the `chrono` / `time`
/// crates which would balloon the trigger's binary size.
///
/// Output looks like `2026-05-19T12:34:56.789Z`.
fn iso8601_utc_now() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs() as i64;
    let millis = now.subsec_millis();

    // Days since 1970-01-01 (Unix epoch).
    let days = secs.div_euclid(86_400);
    let secs_of_day = secs.rem_euclid(86_400) as u32;

    let (year, month, day) = civil_from_days(days);
    let hour = secs_of_day / 3600;
    let minute = (secs_of_day / 60) % 60;
    let second = secs_of_day % 60;

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z",
        year, month, day, hour, minute, second, millis
    )
}

/// Howard Hinnant's `civil_from_days` algorithm — converts days-since-epoch
/// to (year, month, day). Vendored here to dodge the chrono dep.
fn civil_from_days(days: i64) -> (i32, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365; // [0, 399]
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let year = if m <= 2 { y + 1 } else { y } as i32;
    (year, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn civil_from_days_known_values() {
        // 1970-01-01 = day 0
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        // 2000-01-01
        assert_eq!(civil_from_days(10_957), (2000, 1, 1));
        // 2026-05-19 — days_from_epoch verified against Python:
        // (date(2026,5,19) - date(1970,1,1)).days == 20592
        assert_eq!(civil_from_days(20_592), (2026, 5, 19));
    }

    #[test]
    fn iso8601_has_expected_shape() {
        let s = iso8601_utc_now();
        assert_eq!(s.len(), 24);
        assert!(s.ends_with('Z'));
        assert_eq!(&s[4..5], "-");
        assert_eq!(&s[10..11], "T");
        assert_eq!(&s[19..20], ".");
    }

    #[test]
    fn base64_encode_basic() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_encode(b"hello world"), "aGVsbG8gd29ybGQ=");
    }

    #[test]
    fn base64_encode_handles_high_bytes() {
        // Bytes 0xFE 0xFF 0x00 - non-printable, full range.
        let s = base64_encode(&[0xFE, 0xFF, 0x00]);
        // Decoded back should yield the same bytes; sanity-check shape.
        assert_eq!(s.len(), 4);
        assert!(!s.contains('='));
    }
}
