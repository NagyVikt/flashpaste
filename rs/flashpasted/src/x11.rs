//! X11 clipboard ownership.
//!
//! XWayland's CLIPBOARD mirror is the fallback when Mutter wedges the
//! Wayland clipboard (see fact #3 from the spec, and the "wedged data-
//! device" notes in the bash dispatcher).
//!
//! Design:
//!   * One long-lived `x11rb` connection, held forever by a dedicated
//!     blocking thread (spawn_blocking).
//!   * A hidden InputOnly window is the selection owner.
//!   * On every staged-image refresh we call SetSelectionOwner.
//!   * SelectionRequest events arrive on the same connection; we serve the
//!     staged bytes if the requested target is one of TARGETS,
//!     image/png, or image/jpeg.
//!
//! Why a separate thread instead of tokio:
//!   `x11rb::Connection`'s `wait_for_event()` is blocking; integrating it
//!   into tokio's reactor is more complexity than it's worth for an event
//!   stream this sparse. We just park the OS thread on `poll_for_event` /
//!   `wait_for_event`.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use tracing::{debug, error, info, warn};
use x11rb::connection::Connection;
use x11rb::protocol::xproto::{
    AtomEnum, ConnectionExt as _, CreateWindowAux, EventMask, PropMode, SelectionNotifyEvent,
    SelectionRequestEvent, WindowClass, SELECTION_NOTIFY_EVENT,
};
use x11rb::protocol::Event;
use x11rb::rust_connection::RustConnection;
use x11rb::wrapper::ConnectionExt as _;
use x11rb::COPY_DEPTH_FROM_PARENT;

use crate::state::{SharedState, StagedSelection};

pub fn spawn_owner(state: Arc<SharedState>) {
    tokio::task::spawn_blocking(move || {
        if let Err(e) = run(state) {
            error!(error = ?e, "X11 owner exited with error");
        }
    });
}

/// Atom handles we look up once at connection time.
struct Atoms {
    clipboard: u32,
    targets: u32,
    timestamp: u32,
    image_png: u32,
    image_jpeg: u32,
    utf8_string: u32,
    string_atom: u32,
    text_atom: u32,
    text_plain: u32,
    text_plain_utf8: u32,
    multiple: u32,
    atom_pair: u32,
}

impl Atoms {
    fn intern(conn: &RustConnection) -> Result<Self> {
        let clipboard = conn.intern_atom(false, b"CLIPBOARD")?.reply()?.atom;
        let targets = conn.intern_atom(false, b"TARGETS")?.reply()?.atom;
        let timestamp = conn.intern_atom(false, b"TIMESTAMP")?.reply()?.atom;
        let image_png = conn.intern_atom(false, b"image/png")?.reply()?.atom;
        let image_jpeg = conn.intern_atom(false, b"image/jpeg")?.reply()?.atom;
        let utf8_string = conn.intern_atom(false, b"UTF8_STRING")?.reply()?.atom;
        let string_atom = conn.intern_atom(false, b"STRING")?.reply()?.atom;
        let text_atom = conn.intern_atom(false, b"TEXT")?.reply()?.atom;
        let text_plain = conn.intern_atom(false, b"text/plain")?.reply()?.atom;
        let text_plain_utf8 = conn
            .intern_atom(false, b"text/plain;charset=utf-8")?
            .reply()?
            .atom;
        let multiple = conn.intern_atom(false, b"MULTIPLE")?.reply()?.atom;
        let atom_pair = conn.intern_atom(false, b"ATOM_PAIR")?.reply()?.atom;
        Ok(Self {
            clipboard,
            targets,
            timestamp,
            image_png,
            image_jpeg,
            utf8_string,
            string_atom,
            text_atom,
            text_plain,
            text_plain_utf8,
            multiple,
            atom_pair,
        })
    }
}

fn run(state: Arc<SharedState>) -> Result<()> {
    let (conn, screen_num) = match x11rb::connect(None) {
        Ok(c) => c,
        Err(e) => {
            warn!(
                error = %e,
                "X11 connect failed (no DISPLAY or XWayland down); X11 owner disabled"
            );
            return Ok(());
        }
    };

    let screen = &conn.setup().roots[screen_num];
    let root = screen.root;
    let window = conn.generate_id()?;
    // InputOnly hidden window. Listen for property changes (incremental
    // transfer ack from the requestor — we don't currently need INCR but
    // the mask is harmless to set).
    conn.create_window(
        COPY_DEPTH_FROM_PARENT,
        window,
        root,
        -1,
        -1,
        1,
        1,
        0,
        WindowClass::INPUT_ONLY,
        x11rb::NONE,
        &CreateWindowAux::default().event_mask(EventMask::PROPERTY_CHANGE),
    )?;
    // Set a recognizable WM_NAME so xwininfo / xprop debugging shows the
    // owner clearly.
    conn.change_property8(
        PropMode::REPLACE,
        window,
        AtomEnum::WM_NAME,
        AtomEnum::STRING,
        b"flashpasted",
    )?;
    conn.flush()?;

    let atoms = Atoms::intern(&conn)?;
    info!(window, "X11 owner connection up");

    // We need to know "the latest staged image" without locking on every
    // SelectionRequest. The async owner refreshes via SetSelectionOwner +
    // a synchronous `staged_snapshot_blocking()` read inside the handler.
    let mut current_revision: u64 = 0;
    let rx = state.stage_notifier_rx.clone();

    loop {
        // Poll for X events, then check for staging revisions.
        //
        // We can't use `wait_for_event` because then a new screenshot wouldn't
        // trigger a SetSelectionOwner until the next X event arrives. So we
        // poll with a short timeout via `poll_for_event` + sleep, which is
        // ugly but reliable. x11rb 0.13 doesn't expose `wait_for_event_with_
        // sequence_number` in a way that's compatible with tokio without
        // taking a chunky dep.
        match conn.poll_for_event() {
            Ok(Some(event)) => {
                if let Err(e) = handle_event(&conn, window, &atoms, &state, event) {
                    warn!(error = ?e, "X11 event handling error");
                }
            }
            Ok(None) => {
                // No event; check for a refresh.
                let revision = *rx.borrow();
                if revision != current_revision {
                    current_revision = revision;
                    if state.staged_snapshot_blocking().is_some() {
                        // Take ownership of the CLIPBOARD selection. The
                        // CurrentTime (0) sentinel is acceptable here since
                        // we have no recent X timestamp; XWayland mirrors
                        // it through to Mutter regardless.
                        if let Err(e) =
                            conn.set_selection_owner(window, atoms.clipboard, x11rb::CURRENT_TIME)
                        {
                            warn!(error = %e, "set_selection_owner failed");
                        }
                        let _ = conn.flush();
                        debug!(revision, "X11: claimed CLIPBOARD ownership");
                    }
                }
                // ─── We NEVER voluntarily release CLIPBOARD ───────────────
                // A timed `set_selection_owner(NONE)` was tried here as a way
                // to stop a staged screenshot from shadowing the user's text
                // clipboard. It is not survivable: GNOME/Mutter ships no
                // clipboard *manager*, so when the last owner disappears the
                // selection has no owner at all and the system clipboard goes
                // EMPTY — for text as much as for images. Measured on this box:
                // stage 22 bytes of text, wait 8 s, and `xclip -o` returns
                // "target STRING not available" / `wl-paste` returns "Nothing
                // is copied". Symptom the user hit: copy something, tab over to
                // a browser (Facebook), Ctrl+V, and paste an empty string.
                //
                // The correct X11 lifecycle is to hold the selection until
                // another client takes it, which arrives as SelectionClear and
                // needs no action from us. Both text-override paths already
                // work without releasing: an XWayland copy claims CLIPBOARD
                // directly (→ SelectionClear), and a native-Wayland copy is
                // picked up by `read_wayland_text_if_present`, which
                // `resolve_paste_intent` probes BEFORE the X11 reader.
                std::thread::sleep(Duration::from_millis(20));
            }
            Err(e) => {
                error!(error = %e, "X11 poll_for_event failed; exiting owner thread");
                return Err(e.into());
            }
        }
    }
}

fn handle_event(
    conn: &RustConnection,
    window: u32,
    atoms: &Atoms,
    state: &Arc<SharedState>,
    event: Event,
) -> Result<()> {
    match event {
        Event::SelectionRequest(req) => handle_selection_request(conn, window, atoms, state, req),
        Event::SelectionClear(clear) => {
            // Someone else claimed CLIPBOARD; that's fine, we'll reclaim on
            // the next staged-image refresh.
            debug!(selection = clear.selection, "X11: SelectionClear received");
            Ok(())
        }
        _ => Ok(()),
    }
}

fn handle_selection_request(
    conn: &RustConnection,
    window: u32,
    atoms: &Atoms,
    state: &Arc<SharedState>,
    req: SelectionRequestEvent,
) -> Result<()> {
    // Build the SelectionNotify we'll send to the requestor. `property == 0`
    // signals "refused" per ICCCM. We set it to the same atom as the
    // request's `property` field on success.
    let mut notify = SelectionNotifyEvent {
        response_type: SELECTION_NOTIFY_EVENT,
        sequence: 0,
        time: req.time,
        requestor: req.requestor,
        selection: req.selection,
        target: req.target,
        property: 0, // refused by default
    };
    // ICCCM says: if `property == NONE` the requestor wants the data back
    // in `target` itself. (Mostly an obsolete pre-ICCCM convention; we
    // honor it anyway.)
    let property = if req.property == x11rb::NONE {
        req.target
    } else {
        req.property
    };

    // Snapshot the staged image once for this request. None means we don't
    // have anything to serve — reply with property=0.
    let Some(staged) = state.staged_snapshot_blocking() else {
        conn.send_event(false, req.requestor, EventMask::NO_EVENT, notify)?;
        conn.flush()?;
        return Ok(());
    };

    let served = serve_target(
        conn,
        window,
        atoms,
        &staged,
        req.requestor,
        property,
        req.target,
    )
    .with_context(|| format!("serve target atom={}", req.target))?;
    if served {
        notify.property = property;
    }
    conn.send_event(false, req.requestor, EventMask::NO_EVENT, notify)?;
    conn.flush()?;
    Ok(())
}

/// Serve `target` to `requestor` on `property`. Returns true if we wrote
/// data (caller signals success); false means refuse.
fn serve_target(
    conn: &RustConnection,
    _window: u32,
    atoms: &Atoms,
    staged: &StagedSelection,
    requestor: u32,
    property: u32,
    target: u32,
) -> Result<bool> {
    if target == atoms.targets {
        // Respond with the list of supported targets. Composition depends
        // on whether we're serving an image or text selection.
        let mut supported = vec![atoms.targets, atoms.timestamp];
        match staged {
            StagedSelection::Image(img) => {
                // Advertise the image's ACTUAL format, plus image/png as a
                // universal alias (Claude Code reads `wl-paste -t image/png`).
                // Never advertise a format the bytes are NOT: a PNG screenshot
                // used to be advertised as image/jpeg too, so a browser that
                // picked image/jpeg got PNG bytes mislabeled as JPEG — the
                // upload failed and the chat sent a doomed empty message next to
                // the image ("Couldn't send"). Mirrors the Wayland owner policy
                // in wayland.rs (img.mime first, image/png alias).
                if img.mime == "image/jpeg" {
                    supported.push(atoms.image_jpeg);
                }
                supported.push(atoms.image_png);
            }
            StagedSelection::Text(_) => {
                supported.push(atoms.utf8_string);
                supported.push(atoms.string_atom);
                supported.push(atoms.text_atom);
                supported.push(atoms.text_plain_utf8);
                supported.push(atoms.text_plain);
            }
        }
        conn.change_property32(
            PropMode::REPLACE,
            requestor,
            property,
            AtomEnum::ATOM,
            &supported,
        )?;
        return Ok(true);
    }
    if target == atoms.timestamp {
        // ICCCM lets us return CurrentTime (0) for the selection's
        // acquisition time. Real X clipboards return the server timestamp
        // captured when ownership was claimed, but XWayland's mirror is
        // tolerant of CurrentTime.
        conn.change_property32(
            PropMode::REPLACE,
            requestor,
            property,
            AtomEnum::INTEGER,
            &[0u32],
        )?;
        return Ok(true);
    }
    if let StagedSelection::Image(img) = staged {
        // image/png is the universal alias we always serve (Claude reads it).
        // image/jpeg is served ONLY when the staged bytes really are JPEG —
        // never hand a requestor JPEG-labeled PNG bytes (see the TARGETS note).
        let serve_png = target == atoms.image_png;
        let serve_jpeg = target == atoms.image_jpeg && img.mime == "image/jpeg";
        if serve_png || serve_jpeg {
            change_property_chunked(conn, requestor, property, target, &img.bytes)?;
            return Ok(true);
        }
    }
    if let StagedSelection::Text(txt) = staged {
        if target == atoms.utf8_string
            || target == atoms.string_atom
            || target == atoms.text_atom
            || target == atoms.text_plain
            || target == atoms.text_plain_utf8
        {
            // Never serve a zero-length text selection. Writing an empty
            // property is a *successful* conversion as far as the requestor is
            // concerned, so it pastes an empty string rather than leaving the
            // target untouched. Every staging path already rejects empty
            // (`handle_stage_text`, `read_clipboard_text_if_present`,
            // `read_wayland_text_if_present`); this is the outgoing-direction
            // last gate, mirroring `looks_like_text` on the incoming one.
            if txt.bytes.is_empty() {
                warn!("X11: refusing to serve an empty text selection");
                return Ok(false);
            }
            change_property_chunked(conn, requestor, property, target, &txt.bytes)?;
            return Ok(true);
        }
    }
    if target == atoms.multiple {
        // MULTIPLE: the requestor wrote a list of (target, property) atom
        // pairs to its property; we walk them and serve each.
        let reply = conn
            // long_length is in 4-byte units; 65536 = 256KB worth of atom pairs,
            // far more than any sane requestor will use.
            .get_property(false, requestor, property, atoms.atom_pair, 0, 65_536)?
            .reply()?;
        if reply.format != 32 {
            return Ok(false);
        }
        let atoms_payload = reply
            .value32()
            .ok_or_else(|| anyhow::anyhow!("MULTIPLE property had wrong format"))?;
        let mut pairs: Vec<u32> = atoms_payload.collect();
        let mut refused_any = false;
        for chunk in pairs.chunks_mut(2) {
            if chunk.len() != 2 {
                continue;
            }
            let sub_target = chunk[0];
            let sub_property = chunk[1];
            // ICCCM forbids nesting MULTIPLE inside MULTIPLE; honouring it
            // would recurse until the stack blows. Treat it as refused.
            let ok = sub_target != atoms.multiple
                && serve_target(
                    conn,
                    _window,
                    atoms,
                    staged,
                    requestor,
                    sub_property,
                    sub_target,
                )
                .unwrap_or(false);
            if !ok {
                // ICCCM 2.6.2: "If the owner fails to convert the target named
                // by an atom in the ATOM_PAIR, it should replace that atom in
                // the property with None." We used to skip this, leaving the
                // pair intact — so a requestor that asked for text while we
                // held an image was told the conversion SUCCEEDED, read the
                // property we never wrote, and pasted an EMPTY string. GTK and
                // Firefox both request via MULTIPLE, which is how this reached
                // a browser compose box.
                chunk[1] = x11rb::NONE;
                refused_any = true;
            }
        }
        if refused_any {
            conn.change_property32(
                PropMode::REPLACE,
                requestor,
                property,
                atoms.atom_pair,
                &pairs,
            )?;
        }
        // The MULTIPLE conversion itself succeeded — we processed the list and
        // reported per-target outcomes inside it. Per-target failures are
        // signalled by the None atoms above, not by refusing the whole request.
        return Ok(true);
    }
    debug!(target, "X11: refusing unknown target");
    Ok(false)
}

/// Split a large byte slice into ChangeProperty(APPEND) calls so we stay
/// well under the X server's max-request-size. For PNG/JPEG screenshots in
/// the 100KB-2MB range this is typically 1-4 chunks.
fn change_property_chunked(
    conn: &RustConnection,
    window: u32,
    property: u32,
    target: u32,
    data: &[u8],
) -> Result<()> {
    // 256KB is comfortably below any X server's max-request-size minus
    // the request header.
    const CHUNK: usize = 256 * 1024;
    let mut mode = PropMode::REPLACE;
    if data.is_empty() {
        conn.change_property8(mode, window, property, target, data)?;
        return Ok(());
    }
    for chunk in data.chunks(CHUNK) {
        conn.change_property8(mode, window, property, target, chunk)?;
        mode = PropMode::APPEND;
    }
    Ok(())
}
