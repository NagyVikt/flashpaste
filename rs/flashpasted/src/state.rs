//! Daemon-wide shared state and configuration.
//!
//! `SharedState` is what every subsystem (inotify, Wayland, X11, IPC, paste
//! dispatch) hangs off. The big-ticket item is `latest_image` — an
//! `RwLock<Option<StagedImage>>` that the inotify watcher writes and the
//! clipboard owners + paste dispatcher read.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8};
use std::sync::Arc;
use std::time::SystemTime;

use anyhow::Result;
use tokio::sync::{watch, RwLock};

use crate::Args;

/// How fresh a staged image must be before the daemon will use it for a
/// paste op. Bash dispatcher uses a 30s file-age gate; the daemon is more
/// generous because once we own the clipboard the image stays valid until
/// something supersedes it. 30 minutes covers the common AFK-then-paste
/// case (took a screenshot, switched away, came back) without the daemon
/// silently punting to bash and surprising the user.
pub const STAGED_IMAGE_TTL_SECS: u64 = 1800;

/// Text selections live as long as someone is owning the clipboard. Match
/// typical desktop UX: a copy is "fresh" for the rest of the session.
pub const STAGED_TEXT_TTL_SECS: u64 = 24 * 3600;

/// Bytes of a staged image we hold in memory. PNG/JPEG range is 100KB-2MB
/// typically; we keep the bytes so X11's SelectionRequest handler can serve
/// them on demand without re-reading the file.
#[derive(Debug, Clone)]
pub struct StagedImage {
    pub bytes: Arc<Vec<u8>>,
    pub mime: &'static str,
    pub path: PathBuf,
    /// Captured-at as a `SystemTime` so subsystems running on `spawn_blocking`
    /// (X11) can compute ages without needing a tokio handle.
    pub captured_at: SystemTime,
}

impl StagedImage {
    /// Returns true if the image is still inside the daemon's freshness window.
    pub fn is_fresh(&self) -> bool {
        match self.captured_at.elapsed() {
            Ok(age) => age.as_secs() <= STAGED_IMAGE_TTL_SECS,
            Err(_) => true, // clock skewed; treat as fresh rather than drop
        }
    }
}

/// Bytes of a staged TEXT selection. v1.19+ — the daemon now owns text on
/// the clipboard too, so `clipboard-set.sh` can route there instead of
/// forking `wl-copy` (which surfaced as a phantom dock entry on every
/// copy). Kept as raw bytes (not String) because tmux can pipe non-UTF8.
#[derive(Debug, Clone)]
pub struct StagedText {
    pub bytes: Arc<Vec<u8>>,
    pub captured_at: SystemTime,
}

impl StagedText {
    pub fn is_fresh(&self) -> bool {
        match self.captured_at.elapsed() {
            Ok(age) => age.as_secs() <= STAGED_TEXT_TTL_SECS,
            Err(_) => true,
        }
    }
}

/// What the daemon is currently advertising on the clipboard. At most one
/// variant is live at a time — copying text clobbers a staged image and
/// vice versa, mirroring how real system clipboards work.
#[derive(Debug, Clone)]
pub enum StagedSelection {
    Image(StagedImage),
    Text(StagedText),
}

impl StagedSelection {
    pub fn is_fresh(&self) -> bool {
        match self {
            Self::Image(img) => img.is_fresh(),
            Self::Text(txt) => txt.is_fresh(),
        }
    }
}

/// Cached kitty IPC protocol version. Populated once at startup; never on
/// the paste hot path.
#[derive(Debug, Clone, Copy)]
pub struct KittyVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl KittyVersion {
    /// The version we ship as fallback when `kitty --version` is missing or
    /// produces something we can't parse. 0.21.0 is the oldest kitty that
    /// implements `send_text` over IPC with the modern envelope; any newer
    /// kitty (we've verified on 0.46.2) accepts the same payload.
    pub const FALLBACK: KittyVersion = KittyVersion {
        major: 0,
        minor: 21,
        patch: 0,
    };
}

/// Static configuration resolved at startup.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct DaemonConfig {
    pub socket_path: PathBuf,
    pub screenshots_dir: Option<PathBuf>,
    pub kitty_version: KittyVersion,
    pub bash_fallback: PathBuf,
    pub tmux_rebind_command: String,
}

impl DaemonConfig {
    pub fn resolve(args: &Args) -> Result<Self> {
        let socket_path = args.socket.clone().unwrap_or_else(default_socket_path);
        let screenshots_dir = args
            .screenshots_dir
            .clone()
            .or_else(default_screenshots_dir);
        let kitty_version = detect_kitty_version();
        let bash_fallback = default_bash_fallback_path();
        let tmux_rebind_command = default_tmux_rebind_command();

        Ok(Self {
            socket_path,
            screenshots_dir,
            kitty_version,
            bash_fallback,
            tmux_rebind_command,
        })
    }
}

/// Per-process shared state. Cheap to clone (everything is `Arc`).
#[allow(dead_code)]
pub struct SharedState {
    pub config: DaemonConfig,
    pub kitty_version: KittyVersion,
    /// The currently staged selection. `None` until something (a screenshot
    /// via inotify, or a text copy via the stage_text IPC) is staged.
    pub latest_selection: RwLock<Option<StagedSelection>>,
    /// Channel notified every time `latest_selection` is updated. Used by
    /// the Wayland + X11 owners to refresh their selection content.
    pub stage_notifier_tx: watch::Sender<u64>,
    pub stage_notifier_rx: watch::Receiver<u64>,
    /// Duplicate-trigger guard. Holds `(unix-epoch-ms, signature)` of the
    /// last `paste` op the daemon accepted, where `signature` is a hash of
    /// `(pane_id, content_bytes)`. A second trigger that carries the SAME
    /// signature inside the debounce window is acknowledged as deduped
    /// instead of dispatching the same payload to the same pane again.
    ///
    /// Why pane+content rather than time-only: on this box a single Ctrl+V
    /// fires TWO independent handlers — kitty's `map ctrl+v` (slow: spawns
    /// `sh` + `tmux display-message`) and tmux's `bind -n C-v` (fast). They
    /// land hundreds of ms apart, so a narrow time-only window let the pair
    /// through as two pastes ("textext"). Keying on pane+content lets the
    /// window widen enough to swallow that spread WITHOUT ever blocking a
    /// genuinely different paste (different pane, or different bytes).
    /// `std::sync::Mutex` keeps the read-compare-store atomic so two truly
    /// simultaneous identical triggers can't both claim; the critical
    /// section is a couple of integer compares and we never await holding it.
    pub last_paste: std::sync::Mutex<Option<(u64, u64)>>,
    /// True while a paste dispatch is in flight. Subsequent paste requests
    /// during that window dedupe immediately (they're absorbed and replayed
    /// once at completion) instead of stacking — without this guard, a
    /// rapid double-click on the right-click → Paste menu would fire two
    /// \026 bytes in quick succession and Claude could interpret them as
    /// two pastes. The window used to be up to 30 s while the v1.23
    /// `wait_for_pane_idle` hold ran; v1.24 dropped the wait, so the
    /// window is now ~10–20 ms (just the dispatch itself) — still useful
    /// for double-click deduplication.
    pub paste_in_flight: AtomicBool,
    /// Count of paste requests that arrived while a dispatch was already
    /// in flight (saturating at u8::MAX). The in-flight task drains this
    /// at completion and re-dispatches ONCE if it was non-zero — so N
    /// presses during one generation collapse to one extra image attach,
    /// not N. (User contract: "I press paste N times → one image lands".
    /// Watcher reported this was the biggest remaining UX issue.)
    pub pending_paste: AtomicU8,
    /// Pane id of the MOST RECENT press absorbed by `pending_paste`. The
    /// replay dispatch uses this so a press to pane B during pane A's
    /// in-flight dispatch goes to B (not A). Without this, the absorbed
    /// press for B was silently lost: watcher caught it as
    /// "absorbed-press pane=%38 → replay pane=%41 (wrong pane)".
    /// `std::sync::Mutex` is fine — the critical section is a `String`
    /// swap, hundreds of nanoseconds; we never await while holding it.
    pub pending_pane: std::sync::Mutex<Option<String>>,
    /// `captured_at` (ms since epoch) of the image that we last asked the
    /// Wayland + X11 owners to claim via the stage notifier. If the next
    /// paste's staged image has the same id we skip the notifier send and
    /// the 8 ms post-reassert sleep — saves ~10 ms on the common "paste
    /// the same screenshot twice" path and avoids the X11
    /// SetSelectionOwner storm watcher flagged.
    pub last_claim_request_image_ms: AtomicU64,
    /// Last paste-time screenshots-dir scan. The scan is a correctness
    /// fallback for GNOME's delayed close-write behavior, but repeated paste
    /// presses should not all walk the directory.
    pub last_screenshot_scan_ms: AtomicU64,
    /// Last live-clipboard image probe. Used by `handle_paste` to bridge
    /// the "browser Copy Image" case (Firefox/Chrome write bytes to the
    /// clipboard with no file write, so the inotify path misses them).
    /// Throttled because each probe forks `xclip` and may also do a
    /// blocking wl-clipboard read.
    pub last_live_image_probe_ms: AtomicU64,
    /// Memo of `pane_id -> agent kind`. The dispatch path classifies the
    /// foreground agent on every paste; for Claude Code panes that forces a
    /// `ps` descendant walk (the snapshot's `current_command` is `node`/`bun`,
    /// never a bare `claude`). The cache keeps the walk to once per agent
    /// lifetime, re-deriving only when the pane's command or pid changes.
    pub agent_cache: crate::agent::AgentCache,
}

impl SharedState {
    pub fn new(config: DaemonConfig, kitty_version: KittyVersion) -> Self {
        let (tx, rx) = watch::channel(0u64);
        Self {
            config,
            kitty_version,
            latest_selection: RwLock::new(None),
            stage_notifier_tx: tx,
            stage_notifier_rx: rx,
            last_paste: std::sync::Mutex::new(None),
            paste_in_flight: AtomicBool::new(false),
            pending_paste: AtomicU8::new(0),
            pending_pane: std::sync::Mutex::new(None),
            last_claim_request_image_ms: AtomicU64::new(0),
            last_screenshot_scan_ms: AtomicU64::new(0),
            last_live_image_probe_ms: AtomicU64::new(0),
            agent_cache: crate::agent::new_cache(),
        }
    }

    /// Replace the staged image and notify subscribers.
    pub async fn set_staged_image(&self, image: StagedImage) {
        self.set_staged_selection(StagedSelection::Image(image))
            .await;
    }

    /// Replace the staged text and notify subscribers. v1.19+ — the path
    /// through `flashpaste-trigger --stage-text` that lets `clipboard-set.sh`
    /// avoid forking `wl-copy`.
    pub async fn set_staged_text(&self, text: StagedText) {
        self.set_staged_selection(StagedSelection::Text(text)).await;
    }

    /// Replace whatever's currently staged with `sel`. Either variant
    /// clobbers the other — clipboards are a single slot.
    pub async fn set_staged_selection(&self, sel: StagedSelection) {
        {
            let mut guard = self.latest_selection.write().await;
            *guard = Some(sel);
        }
        // Best-effort: ignore send errors (no subscribers is fine).
        let _ = self.stage_notifier_tx.send(now_unix_ms());
    }

    /// Snapshot the staged selection (without holding the lock during use).
    pub async fn staged_snapshot(&self) -> Option<StagedSelection> {
        self.latest_selection.read().await.clone()
    }

    /// Image-only convenience for the paste op (which is image-specific).
    /// Returns `None` if nothing is staged OR if the staged item is text.
    #[allow(dead_code)]
    pub async fn staged_image(&self) -> Option<StagedImage> {
        match self.staged_snapshot().await {
            Some(StagedSelection::Image(img)) => Some(img),
            _ => None,
        }
    }

    /// Synchronous variant for subsystems that aren't tokio-aware (e.g. the
    /// X11 owner running on `spawn_blocking`). Uses `blocking_read` because
    /// we are guaranteed to be inside a blocking task.
    pub fn staged_snapshot_blocking(&self) -> Option<StagedSelection> {
        self.latest_selection.blocking_read().clone()
    }
}

fn default_socket_path() -> PathBuf {
    if let Ok(dir) = std::env::var("XDG_RUNTIME_DIR") {
        if !dir.is_empty() {
            return PathBuf::from(dir).join("flashpaste.sock");
        }
    }
    let uid = nix::unistd::Uid::current().as_raw();
    let candidate = PathBuf::from(format!("/run/user/{uid}"));
    if candidate.is_dir() {
        return candidate.join("flashpaste.sock");
    }
    PathBuf::from("/tmp").join("flashpaste.sock")
}

fn default_screenshots_dir() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    Some(PathBuf::from(home).join("Pictures").join("Screenshots"))
}

/// Mirrors the bash dispatcher's rebind line so a daemon-handled paste
/// leaves tmux in the same state as the legacy script's:
///   tmux bind -n C-v run-shell -b "TMUX_PASTE_TRIGGER=ctrl-v <trigger> '#{pane_id}'"
///
/// IMPORTANT: this points at the bash dispatcher, not the trigger binary,
/// to keep the daemon-failure recovery path obvious. If the daemon is
/// running, the FIRST paste sets the binding via this string AND the trigger
/// short-circuits anyway. If the user wants the rebind to point at
/// `flashpaste-trigger` instead, they edit `examples/tmux.conf.snippet`.
fn default_tmux_rebind_command() -> String {
    // Match the user's tmux.conf binding exactly: prefer the daemon trigger
    // and fall back to the bash dispatcher via `||`. Before this, the
    // default rebound to bash-only after every paste, silently demoting
    // Tier 3 → Tier 1 for the rest of the tmux session (until the user
    // re-sourced ~/.tmux.conf).
    String::from(
        "tmux bind -n C-v run-shell -b \"TMUX_PASTE_TRIGGER=ctrl-v \
         flashpaste-trigger '#{pane_id}' 2>/dev/null || \
         TMUX_PASTE_TRIGGER=ctrl-v tmux-paste-dispatch '#{pane_id}' 2>/dev/null || \
         TMUX_PASTE_TRIGGER=ctrl-v tmux-paste-dispatch.sh '#{pane_id}'\"",
    )
}

fn default_bash_fallback_path() -> PathBuf {
    if let Ok(path) = std::env::var("FLASHPASTE_BASH_FALLBACK") {
        if !path.is_empty() {
            return PathBuf::from(path);
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home)
            .join(".local")
            .join("bin")
            .join("tmux-paste-dispatch.sh");
    }
    PathBuf::from("/usr/bin/tmux-paste-dispatch")
}

/// Detect the kitty IPC protocol version at startup. We run `kitty --version`
/// ONCE. This is process-startup, not paste hot-path, so the cost is free.
fn detect_kitty_version() -> KittyVersion {
    let output = std::process::Command::new("kitty")
        .arg("--version")
        .output();
    match output {
        Ok(out) if out.status.success() => {
            let s = String::from_utf8_lossy(&out.stdout);
            parse_kitty_version(&s).unwrap_or(KittyVersion::FALLBACK)
        }
        _ => KittyVersion::FALLBACK,
    }
}

fn parse_kitty_version(s: &str) -> Option<KittyVersion> {
    // `kitty 0.46.2 created by Kovid Goyal` — second whitespace-delimited
    // token is the dotted version.
    let token = s.split_whitespace().nth(1)?;
    let mut parts = token.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    let patch = parts.next().unwrap_or("0").parse().ok()?;
    Some(KittyVersion {
        major,
        minor,
        patch,
    })
}

pub fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_real_kitty_output() {
        let v = parse_kitty_version("kitty 0.46.2 created by Kovid Goyal").unwrap();
        assert_eq!(v.major, 0);
        assert_eq!(v.minor, 46);
        assert_eq!(v.patch, 2);
    }

    #[test]
    fn parses_two_segment_version() {
        let v = parse_kitty_version("kitty 0.42 something else").unwrap();
        assert_eq!(v.minor, 42);
        assert_eq!(v.patch, 0);
    }

    #[test]
    fn rejects_garbage() {
        assert!(parse_kitty_version("").is_none());
        assert!(parse_kitty_version("not a version").is_none());
    }
}
