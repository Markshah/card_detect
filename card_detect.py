#!/usr/bin/env python3
# card_detect.py — face-up/down by white rim + arming/reset + dual-WS sends + static console dashboard
import os, sys, cv2, time, json, atexit, numpy as np, logging
from collections import deque
from dotenv import load_dotenv

# ---------------- env ----------------
if os.path.exists("env"): load_dotenv("env")
def _b(k, d="0"): return os.getenv(k, d).strip().lower() in ("1","true","yes")

def _get_csv_ints(name, default="0,0,0,0"):
    raw = os.getenv(name, default)
    raw = raw.split("#", 1)[0].strip().strip('"').strip("'")
    parts = [p.strip() for p in raw.split(",")]
    return tuple(int(p or "0") for p in parts[:4]) if len(parts) >= 4 else (0,0,0,0)

QUIET_LOGS = _b("QUIET_LOGS", "1")  # keeps dashboard static by muting noisy INFO logs
DASH_ROWS  = int(os.getenv("DASH_ROWS", "6"))  # recent event lines to keep

# ---- camera / I/O ----
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
ROI          = _get_csv_ints("ROI", "0,0,0,0")  # x,y,w,h; 0s = full frame
SLEEP_SEC    = float(os.getenv("FIXED_LOOP_INTERVAL_SECONDS", "1.0"))
SAVE_DIR     = os.getenv("SAVE_FRAME_DIR", "./frames")
DEBUG_DIR    = os.getenv("DEBUG_DUMP_DIR", "./debug")
DRAW_STATS   = _b("DRAW_LABEL_STATS","1")
WARP_SIZE    = (int(os.getenv("WARP_W","400")), int(os.getenv("WARP_H","560")))

SAVE_IMAGES = int(os.getenv("SAVE_IMAGES", "0"))   # 0=off, 1=on
SAVE_WARPS  = int(os.getenv("SAVE_WARPS", "0"))    # 0=off, 1=on
SAVE_GRAYSCALE = int(os.getenv("SAVE_GRAYSCALE", "0"))  # 0=off, 1=save grayscale frames

# ---- rim classifier knobs (env-tunable) ----
RIM_OUTER_FRAC   = float(os.getenv("RIM_OUTER_FRAC", "0.08"))
RIM_INNER_FRAC   = float(os.getenv("RIM_INNER_FRAC", "0.30"))
CHROMA_MAX       = float(os.getenv("WHITE_CHROMA_MAX", "12"))
REL_WHITE_PCT    = float(os.getenv("REL_WHITE_PCT", "0.60"))
FACEUP_WHITE_MIN = float(os.getenv("FACEUP_WHITE_MIN", "0.20"))
RIM_EDGE_MAX     = float(os.getenv("RIM_EDGE_MAX", "0.09"))
CENTER_EDGE_MAX  = float(os.getenv("CENTER_EDGE_MAX","0.28"))

# ---- candidate size/shape guards ----
CARD_AABB_MIN_FRAC = float(os.getenv("CARD_AABB_MIN_FRAC","0.0018"))
CARD_AABB_MAX_FRAC = float(os.getenv("CARD_AABB_MAX_FRAC","0.30"))
CARD_SHORTSIDE_MIN = int(os.getenv("CARD_SHORTSIDE_MIN","90"))
ASPECT_MIN, ASPECT_MAX = float(os.getenv("ASPECT_MIN","1.25")), float(os.getenv("ASPECT_MAX","1.75"))
SOLIDITY_MIN = float(os.getenv("SOLIDITY_MIN","0.88"))
FILL_OBB_MIN = float(os.getenv("FILL_OBB_MIN","0.80"))
ANGLE_TOL_DEG= float(os.getenv("ANGLE_TOL_DEG","16"))

# absolute pixel gates (stable in a fixed rig)
CARD_SHORT_MIN_PX = int(os.getenv("CARD_SHORT_MIN_PX", "130"))
CARD_SHORT_MAX_PX = int(os.getenv("CARD_SHORT_MAX_PX", "260"))
CARD_LONG_MIN_PX  = int(os.getenv("CARD_LONG_MIN_PX",  "180"))
CARD_LONG_MAX_PX  = int(os.getenv("CARD_LONG_MAX_PX",  "380"))

# ---- arming / reset policy ----
ARM_CONSEC_N       = int(os.getenv("ARM_CONSEC_N", "3"))
ARM_FACEUP_MIN     = int(os.getenv("ARM_FACEUP_MIN", "2"))
ZERO_UP_CONSEC_N   = int(os.getenv("ZERO_UP_CONSEC_N", "3"))  # reset only after N consecutive zero-up loops

# send policy
RESEND_EVERY       = int(os.getenv("RESEND_EVERY", "5"))  # periodic same-value push
USE_GRAYSCALE_ONLY = int(os.getenv("USE_GRAYSCALE_ONLY","1"))

# ---- websocket (DUAL DESTINATIONS) ----
from ws_mgr import WSManager
TABLET_WS_URL  = os.getenv("TABLET_WS_URL",  "ws://192.168.1.246:8765").strip()  # cards_detected
ARDUINO_WS_URL = os.getenv("ARDUINO_WS_URL", "ws://192.168.1.245:8888").strip()  # move_dealer_forward

# burst/timeout knobs for Arduino reset command
WS_BURST_SENDS      = int(os.getenv("WS_BURST_SENDS", "1"))
WS_BURST_SPACING_MS = int(os.getenv("WS_BURST_SPACING_MS", "40"))
WS_SEND_RETRIES     = int(os.getenv("WS_SEND_RETRIES", "4"))
WS_RETRY_DELAY_MS   = int(os.getenv("WS_RETRY_DELAY_MS", "150"))
WS_AWAIT_CONNECT_S  = float(os.getenv("WS_AWAIT_CONNECT_SEC", "1.5"))
RESET_DEBOUNCE_SEC  = float(os.getenv("RESET_DEBOUNCE_SEC", "1.2"))

# ---- logging quiet mode ----
if QUIET_LOGS:
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("websocket").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

# Two persistent sockets
ws_tablet  = WSManager(url=TABLET_WS_URL)
ws_arduino = WSManager(url=ARDUINO_WS_URL)
ws_tablet.start()
ws_arduino.start()
atexit.register(lambda: (ws_tablet.stop(), ws_arduino.stop()))

# ---------------- dashboard helpers ----------------
events = deque(maxlen=DASH_ROWS)

def log_event(s: str):
    ts = time.strftime("%H:%M:%S")
    events.appendleft(f"{ts}  {s}")


def render_dashboard(face_up_now, cards_now, armed, arm_streak, zero_up_streak, peak_up):
    import sys
    sys.stdout.write("\033[2J\033[H")  # clear + home

    # ANSI colors
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    BLUE   = "\033[1;34m"
    GREEN  = "\033[1;32m"
    RED    = "\033[1;31m"
    YELLOW = "\033[1;33m"
    CYAN   = "\033[36m"
    MAG    = "\033[35m"
    GRAY   = "\033[90m"

    # Header
    print(f"{BOLD}{CYAN}=== WED NIGHT POKER — CARD DETECT (static console) ==={RESET}")

    # Live WS status
    t_ok = ws_tablet.is_connected
    a_ok = ws_arduino.is_connected
    t_color = GREEN if t_ok else RED
    a_color = GREEN if a_ok else RED
    print(f"{BOLD}[WS]{RESET} Tablet : {t_color}{TABLET_WS_URL}{RESET}   [{t_color}{'UP' if t_ok else 'DOWN'}{RESET}]")
    print(f"{BOLD}[WS]{RESET} Arduino: {a_color}{ARDUINO_WS_URL}{RESET}   [{a_color}{'UP' if a_ok else 'DOWN'}{RESET}]")

    # Config snapshot
    print(f"{GRAY}[CFG]{RESET} CAMERA={CAMERA_INDEX} ROI={ROI} LOOP={SLEEP_SEC:.2f}s GRAY_ONLY={USE_GRAYSCALE_ONLY}")

    # Live state (ARMED green / RESET yellow / IDLE red)
    if armed:
        state_txt = f"{GREEN}ARMED{RESET}"
    elif zero_up_streak >= ZERO_UP_CONSEC_N:
        state_txt = f"{YELLOW}RESET{RESET}"
    else:
        state_txt = f"{RED}IDLE{RESET}"

    down_now = max(0, cards_now - face_up_now)
    print(
        f"{BOLD}[LIVE]{RESET} cards={CYAN}{cards_now}{RESET} "
        f"up={CYAN}{face_up_now}{RESET} down={GRAY}{down_now}{RESET}  "
        f"state={state_txt}  "
        f"arm_streak={arm_streak}/{ARM_CONSEC_N}  "
        f"zero_streak={zero_up_streak}/{ZERO_UP_CONSEC_N}  "
        f"peak_up={peak_up}"
    )

    # Divider + events
    print(f"{GRAY}" + "-" * 72 + f"{RESET}")
    print("Recent events:")
    if events:
        for line in list(events):
            if "ARMED" in line:
                color = GREEN
            elif "RESET" in line:
                color = YELLOW
            elif "failed" in line or "error" in line.lower():
                color = RED
            else:
                color = GRAY
            print("  " + color + line + RESET)
    else:
        print("  (none)")

    sys.stdout.flush()


def _startup_summary():
    # give each connection a moment to come up
    t_ok = ws_tablet.wait_connected(2.0)
    a_ok = ws_arduino.wait_connected(2.0)
    log_event(f"Tablet {'connected' if t_ok else 'pending'}")
    log_event(f"Arduino {'connected' if a_ok else 'pending'}")

def print_ws_status():
    """Display WebSocket connection statuses in color."""
    reset = "\033[0m"
    green = "\033[1;32m"
    red   = "\033[1;31m"
    bold  = "\033[1m"

    t_ok = ws_tablet.is_connected
    a_ok = ws_arduino.is_connected

    t_color = green if t_ok else red
    a_color = green if a_ok else red

    print(f"{bold}[WS]{reset} tablet : {t_color}{TABLET_WS_URL}{reset}   [{t_color}{'UP' if t_ok else 'DOWN'}{reset}]")
    print(f"{bold}[WS]{reset} arduino: {a_color}{ARDUINO_WS_URL}{reset}   [{a_color}{'UP' if a_ok else 'DOWN'}{reset}]")

# -------------- vision helpers --------------
def _roi(frame, r):
    x,y,w,h = r
    if w <= 0 or h <= 0: return frame, 0, 0
    x2, y2 = min(frame.shape[1], x+w), min(frame.shape[0], y+h)
    return frame[y:y2, x:x2].copy(), x, y

def _order_quad(pts):
    s = pts.sum(axis=1); d = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]; br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]; bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def _right_angle_quad(quad, tol=ANGLE_TOL_DEG):
    p = np.array(quad, dtype=np.float32)
    v = np.roll(p, -1, axis=0) - p
    for i in range(4):
        a = v[i-1]; b = v[i]
        c = np.degrees(np.arccos(
            np.clip(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-6), -1, 1)
        ))
        if abs(c-90) > tol: return False
    return True

def _normalize_L(gray):
    lo, hi = np.percentile(gray, (1,99)); span = max(1, hi - lo)
    return np.clip((gray - lo) / span * 255, 0, 255).astype(np.uint8)

def _warp_from_contour(img_bgr, cnt, out_size=WARP_SIZE):
    rect = cv2.minAreaRect(cnt)
    box  = cv2.boxPoints(rect).astype(np.float32)
    W,H = max(int(rect[1][0]),1), max(int(rect[1][1]),1)
    ow, oh = out_size
    if W > H: ow, oh = oh, ow  # landscape
    src = _order_quad(box)
    dst = np.array([[0,0],[ow-1,0],[ow-1,oh-1],[0,oh-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warp = cv2.warpPerspective(img_bgr, M, (ow, oh))
    return warp, box, (W,H)

def _card_candidates(mask, frame_w, frame_h):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 3000: continue
        x,y,w,h = cv2.boundingRect(cnt)
        aabb_frac = (w*h)/float(frame_w*frame_h + 1e-6)
        if not (CARD_AABB_MIN_FRAC <= aabb_frac <= CARD_AABB_MAX_FRAC): continue
        if min(w,h) < CARD_SHORTSIDE_MIN: continue

        hull = cv2.convexHull(cnt)
        solidity = area / (cv2.contourArea(hull)+1e-6)
        (cx,cy),(mw,mh),ang = cv2.minAreaRect(cnt)
        asp = max(mw,mh)/max(1.0,min(mw,mh))
        obb_fill = area / (mw*mh + 1e-6)
        if not (ASPECT_MIN <= asp <= ASPECT_MAX): continue
        if solidity < SOLIDITY_MIN or obb_fill < FILL_OBB_MIN: continue

        quad = cv2.boxPoints(cv2.minAreaRect(cnt))
        if not _right_angle_quad(quad): continue

        # absolute pixel size guard
        short_px = min(mw, mh); long_px = max(mw, mh)
        if not (CARD_SHORT_MIN_PX <= short_px <= CARD_SHORT_MAX_PX and
                CARD_LONG_MIN_PX  <= long_px  <= CARD_LONG_MAX_PX):
            continue

        yield cnt

# -------------- ring-only classifier (with optional debug panel) --------------
def classify_by_white_rim(warp_bgr, make_panel=True):
    lab = cv2.cvtColor(warp_bgr, cv2.COLOR_BGR2LAB)
    L   = lab[...,0].astype(np.float32)
    a   = lab[...,1].astype(np.float32) - 128.0
    b   = lab[...,2].astype(np.float32) - 128.0
    C   = np.hypot(a, b)

    h, w = L.shape
    rim   = int(max(1, RIM_OUTER_FRAC * min(h, w)))
    inner = int(max(rim+1, RIM_INNER_FRAC * min(h, w)))

    band = np.zeros_like(L, dtype=bool)
    band[:inner,:]  = True;  band[-inner:,:]  = True
    band[:, :inner] = True;  band[:, -inner:] = True
    inner_cut = np.zeros_like(L, dtype=bool)
    inner_cut[:rim,:]  = True; inner_cut[-rim:,:]  = True
    inner_cut[:, :rim] = True; inner_cut[:, -rim:] = True
    ring = band & (~inner_cut)

    L_ring = L[ring]
    if L_ring.size == 0:
        return "face_down", {"rule":"no_ring"}, None

    L_thr = np.percentile(L_ring, REL_WHITE_PCT * 100.0)

    if USE_GRAYSCALE_ONLY:
        white_mask = (L >= L_thr) & ring
    else:
        white_mask = (L >= L_thr) & (C <= CHROMA_MAX) & ring

    white_frac = float(white_mask.mean())

    normL = _normalize_L(L.astype(np.uint8))
    edges = cv2.Canny(normL, 30, 90)
    rim_edge = float((edges[ring] > 0).mean())

    csz = max(10, min(w, h)//5)
    cx0 = w//2 - csz//2; cy0 = h//2 - csz//2
    center_edge = float((edges[cy0:cy0+csz, cx0:cx0+csz] > 0).mean())

    stats = {
        "crm": round(white_frac,3),
        "rim_e": round(rim_edge,3),
        "center_ed": round(center_edge,3),
        "L_thr": float(L_thr)
    }

    label = "face_up" if (white_frac >= FACEUP_WHITE_MIN and rim_edge <= RIM_EDGE_MAX and center_edge <= CENTER_EDGE_MAX) \
            else ("face_down" if (white_frac <= 0.20 or rim_edge >= RIM_EDGE_MAX+0.02) else "face_down")

    panel = None
    if SAVE_WARPS and make_panel:
        overlay = warp_bgr.copy()
        ring_vis = np.zeros_like(overlay)
        ring_vis[white_mask] = (0,255,0)
        dbg1 = cv2.addWeighted(overlay, 0.85, ring_vis, 0.45, 0)
        white_img = (white_mask.astype(np.uint8)*255)
        edges_ring = (edges * ring.astype(np.uint8))
        white_bgr = cv2.cvtColor(white_img, cv2.COLOR_GRAY2BGR)
        edges_bgr = cv2.cvtColor(edges_ring, cv2.COLOR_GRAY2BGR)
        panel = np.hstack([dbg1, white_bgr, edges_bgr])

    return label, {"rule":"rim" if label=="face_up" else "rim-reject", **stats}, panel

# -------------- WS helpers --------------
_last_obs = None
_same_streak = 0
_last_reset_ts = 0.0

def send_cards_change_or_every(count: int):
    """Send {'command':'cards_detected'} on change or every RESEND_EVERY identical reads (to TABLET)."""
    global _last_obs, _same_streak
    count = int(count)
    if _last_obs is None or count != _last_obs:
        _last_obs = count; _same_streak = 1
        ws_tablet.send_cards_detected(count)
        log_event(f"cards_detected -> {count} (changed)")
        return True
    _same_streak += 1
    if _same_streak >= RESEND_EVERY:
        _same_streak = 0
        ws_tablet.send_cards_detected(count)
        log_event(f"cards_detected -> {count} (periodic)")
        return True
    return False

def send_reset_reliably():
    """Debounced, retried, bursty reset → move_dealer_forward (to ARDUINO)."""
    global _last_reset_ts
    now = time.time()
    if now - _last_reset_ts < RESET_DEBOUNCE_SEC:
        log_event(f"reset skipped (debounce {now - _last_reset_ts:.1f}s)")
        return False

    ws_arduino.wait_connected(WS_AWAIT_CONNECT_S)
    ok = False
    for attempt in range(1, WS_SEND_RETRIES+1):
        ok = ws_arduino.send_move_dealer_forward_burst(
            burst=WS_BURST_SENDS, spacing_ms=WS_BURST_SPACING_MS
        )
        if ok:
            log_event(f"reset sent (attempt {attempt}/{WS_SEND_RETRIES})")
            break
        log_event(f"reset retry {attempt}/{WS_SEND_RETRIES} in {WS_RETRY_DELAY_MS}ms")
        time.sleep(WS_RETRY_DELAY_MS/1000.0)

    if ok: _last_reset_ts = time.time()
    else:  log_event("reset FAILED after retries")
    return ok

# -------------- main --------------
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] camera not found")
        return

    _startup_summary()

    log_event("loop starting")

    # streak state
    armed = False
    arm_streak = 0
    zero_up_streak = 0
    peak_up = 0

    while True:
        ts = time.strftime("%Y%m%d-%H%M%S")
        ok, frame = cap.read()
        if not ok:
            time.sleep(SLEEP_SEC)
            render_dashboard(0, 0, armed, arm_streak, zero_up_streak, peak_up)
            continue

        proc, offx, offy = _roi(frame, ROI) if sum(ROI) != 0 else (frame, 0, 0)

        gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        norm = _normalize_L(gray)
        _, th = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        cards = []
        for idx, cnt in enumerate(_card_candidates(th, proc.shape[1], proc.shape[0])):
            warp, box, (mw, mh) = _warp_from_contour(proc, cnt)
            label, info, panel = classify_by_white_rim(warp)
            cards.append((label, info, box, panel))

        face_up_now = sum(1 for c in cards if c[0] == "face_up")
        cards_now   = len(cards)
        peak_up     = max(peak_up, face_up_now)

        # --- arming/reset state machine ---
        if not armed:
            if face_up_now >= ARM_FACEUP_MIN:
                arm_streak += 1
            else:
                arm_streak = 0
            if arm_streak >= ARM_CONSEC_N:
                armed = True
                send_cards_change_or_every(face_up_now)
                zero_up_streak = 0
                peak_up = face_up_now
                log_event(f"STATE -> ARMED (need {ARM_FACEUP_MIN}+ for {ARM_CONSEC_N})")
        else:
            if face_up_now == 0:
                zero_up_streak += 1
            else:
                zero_up_streak = 0

            if zero_up_streak >= ZERO_UP_CONSEC_N:
                log_event(f"STATE -> RESET (zero-up for {ZERO_UP_CONSEC_N}, peak={peak_up})")
                send_cards_change_or_every(0)
                send_reset_reliably()
                armed = False
                arm_streak = 0
                zero_up_streak = 0
                peak_up = 0
            else:
                send_cards_change_or_every(face_up_now)

        # annotate & optional saves
        for idx, (label, info, box, panel) in enumerate(cards):
            color = (0,255,0) if label=="face_up" else (0,0,255)
            box = (box + np.array([[offx,offy]], dtype=np.float32)).astype(int)
            cv2.polylines(frame, [box], True, color, 2)
            cv2.putText(frame, f"{idx}:{label}", tuple(box[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if DRAW_STATS:
                s = f"crm={info.get('crm',0):.3f} rim_e={info.get('rim_e',0):.3f} c={info.get('center_ed',0):.3f}"
                cv2.putText(frame, s, (box[0][0], box[0][1]+18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            if SAVE_WARPS and panel is not None:
                fn = os.path.join(DEBUG_DIR, f"{ts}_card{idx}_{label}_crm{info['crm']}_re{info['rim_e']}.png")
                cv2.imwrite(fn, panel)

        if SAVE_IMAGES:
            cv2.imwrite(os.path.join(SAVE_DIR, f"frame_{ts}.jpg"), frame)

        if SAVE_GRAYSCALE:
            gray_path = os.path.join(SAVE_DIR, f"gray_{ts}.jpg")
            cv2.imwrite(gray_path, norm)

        # draw dashboard last
        render_dashboard(face_up_now, cards_now, armed, arm_streak, zero_up_streak, peak_up)
        time.sleep(SLEEP_SEC)

if __name__ == "__main__":
    main()

