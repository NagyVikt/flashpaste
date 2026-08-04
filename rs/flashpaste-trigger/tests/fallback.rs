//! Regression tests for the trigger's "should I run the bash dispatcher?"
//! decision — the one place where a single Ctrl+V can turn into TWO pastes.
//!
//! The bug these pin down: the daemon replies only *after* it has finished
//! dispatching, so an image paste's reply carries the full dispatch cost
//! (measured on a real box: median 37 ms, p90 242 ms, max 341 ms). With the
//! old 150 ms read timeout, 20% of image pastes timed out here, were reported
//! as a daemon *error*, and exec'd `tmux-paste-dispatch.sh` — which pasted the
//! same screenshot a second time, ~330 ms after the daemon's own paste. The
//! daemon's `(pane, content)` dedup cannot see that second paste: it never
//! crosses the socket.
//!
//! The rule under test: once the request is in a LIVE daemon's hands, a
//! missing reply must never trigger the bash fallback. A daemon that stopped
//! answering entirely still must.

use std::io::{Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

/// How the fake daemon answers a `paste` request.
#[derive(Clone, Copy, PartialEq)]
enum Mode {
    /// Alive and answering `ping`, but the paste reply lands well after the
    /// trigger's read timeout. This is the real double-paste scenario.
    Slow,
    /// Accepts connections and never answers anything, `ping` included.
    Wedged,
    /// Answers immediately with `ok:false` — the daemon's legitimate punt.
    Decline,
}

struct FakeDaemon {
    dir: PathBuf,
}

impl Drop for FakeDaemon {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

impl FakeDaemon {
    /// AF_UNIX paths are capped near 108 bytes, so the socket lives in a
    /// short `/tmp` dir rather than under `target/`.
    fn start(name: &str, mode: Mode) -> Self {
        let dir = PathBuf::from(format!("/tmp/fp-trig-{}-{}", std::process::id(), name));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create test runtime dir");

        let listener = UnixListener::bind(dir.join("flashpaste.sock")).expect("bind fake daemon");
        std::thread::spawn(move || {
            for conn in listener.incoming().flatten() {
                std::thread::spawn(move || serve(conn, mode));
            }
        });
        Self { dir }
    }

    /// Run the trigger against this daemon. Returns `true` when the bash
    /// fallback ran (i.e. a second paste would have happened).
    fn run_trigger(&self) -> (bool, Option<i32>) {
        let marker = self.dir.join("fallback-ran");
        let script = self.dir.join("fake-dispatch.sh");
        std::fs::write(
            &script,
            format!("#!/bin/sh\necho ran >> {}\n", marker.display()),
        )
        .expect("write fake dispatcher");
        set_executable(&script);

        let status = Command::new(env!("CARGO_BIN_EXE_flashpaste-trigger"))
            .arg("%99")
            .env("XDG_RUNTIME_DIR", &self.dir)
            .env("FLASHPASTE_BASH_FALLBACK", &script)
            .env("TMUX_PASTE_TRIGGER", "ctrl-v")
            .env("FLASHPASTE_QUIET", "1")
            .status()
            .expect("spawn trigger");
        (marker.exists(), status.code())
    }
}

fn set_executable(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let mut perms = std::fs::metadata(path).expect("stat script").permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(path, perms).expect("chmod script");
}

fn serve(mut conn: UnixStream, mode: Mode) {
    let Some(req) = read_frame(&mut conn) else {
        return;
    };
    if mode == Mode::Wedged {
        // Never answer — not the paste, not the liveness ping.
        std::thread::sleep(Duration::from_secs(30));
        return;
    }
    let is_ping = String::from_utf8_lossy(&req).contains("\"ping\"");
    if is_ping {
        write_frame(&mut conn, br#"{"ok":true,"pong":true}"#);
        return;
    }
    match mode {
        // Longer than the trigger's read timeout, so the reply is guaranteed
        // to arrive too late — exactly what a slow image dispatch looks like.
        Mode::Slow => {
            std::thread::sleep(Duration::from_millis(3000));
            write_frame(&mut conn, br#"{"ok":true,"kind":"image"}"#);
        }
        Mode::Decline => write_frame(&mut conn, br#"{"ok":false,"fallback":"bash"}"#),
        Mode::Wedged => unreachable!("handled above"),
    }
}

fn read_frame(conn: &mut UnixStream) -> Option<Vec<u8>> {
    let mut len_buf = [0u8; 4];
    conn.read_exact(&mut len_buf).ok()?;
    let mut body = vec![0u8; u32::from_le_bytes(len_buf) as usize];
    conn.read_exact(&mut body).ok()?;
    Some(body)
}

fn write_frame(conn: &mut UnixStream, body: &[u8]) {
    let len = body.len() as u32;
    let _ = conn.write_all(&len.to_le_bytes());
    let _ = conn.write_all(body);
}

#[test]
fn slow_daemon_does_not_run_the_bash_fallback() {
    // THE regression. The daemon is alive and pasting; it is just slower than
    // our read timeout. Running the bash dispatcher here is the double paste.
    let daemon = FakeDaemon::start("slow", Mode::Slow);
    let (fallback_ran, code) = daemon.run_trigger();
    assert!(
        !fallback_ran,
        "bash fallback ran while the daemon was alive and mid-paste — that is the double paste"
    );
    // Exit 0 matters as much as skipping the exec: tmux runs the trigger as
    // `flashpaste-trigger ... || tmux-paste-dispatch.sh`, so a non-zero exit
    // would let the shell fire the second paste behind our back.
    assert_eq!(
        code,
        Some(0),
        "must exit 0 so tmux's `||` fallback stays put"
    );
}

#[test]
fn wedged_daemon_still_falls_back_to_bash() {
    // The safety the timeout was there for in the first place: a daemon that
    // answers nothing at all must not leave the user without a paste.
    let daemon = FakeDaemon::start("wedged", Mode::Wedged);
    let (fallback_ran, _) = daemon.run_trigger();
    assert!(
        fallback_ran,
        "a daemon that answers nothing (not even ping) must fall back to bash"
    );
}

#[test]
fn declined_paste_still_falls_back_to_bash() {
    // `{"ok":false,"fallback":"bash"}` is the daemon's deliberate punt
    // (nothing staged, stale image). It must reach bash unchanged.
    let daemon = FakeDaemon::start("decline", Mode::Decline);
    let (fallback_ran, _) = daemon.run_trigger();
    assert!(fallback_ran, "an explicit daemon punt must reach bash");
}
