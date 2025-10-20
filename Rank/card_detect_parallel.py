#!/usr/bin/env python3
# Wednesday Night Poker — Card Detector
# v2025.10.19
# Features: 5-slot tracker (flop/turn/river), 5-threaded classifier pool, SHA-1 digest tracking, early tablet updates

# card_detect.py — face-up/down by white rim + arming/reset + dual-WS sends + dashboard
import os, sys, cv2, time, atexit, numpy as np, logging, hashlib, threading, queue
from collections import deque
from dotenv import load_dotenv
from pyfiglet import Figlet
from rank_suit_parallel import classify_fullcard_anyrot as _classify_card  # returns (code, score, rot)

# ---- OpenCV perf knobs ( #4 ) ----
try:
    cv2.setUseOptimized(True)
    # 0 lets OpenCV pick; set a small fixed number (e.g., 4) if your CPU likes that better
    cv2.setNumThreads(0)
except Exception:
    pass


# ---------------- env ----------------
if os.path.exists("env"): load_dotenv("env")
def _b(k, d="0"): return os.getenv(k, d).strip().lower() in ("1","true","yes")

# --------- simple "-KEY value" CLI parser ---------
def parse_dash_args(argv):
    args = {}
    key = None
    for token in argv:
        if token.startswith('-') and not token.startswith('--'):
            key = token.lstrip('-'); args[key] = None
        else:
            if key: args[key] = token; key = None
    return args

_cli = parse_dash_args(sys.argv[1:])
if _cli:
    print("[CLI overrides detected]", _cli)
for k, v in _cli.items(): os.environ[k] = v

def _get_csv_ints(name, default="0,0,0,0"):
    raw = os.getenv(name, default)
    raw = raw.split("#", 1)[0].strip().strip('"').strip("'")
    parts = [p.strip() for p in raw.split(",")]
    return tuple(int(p or "0") for p in parts[:4]) if len(parts) >= 4 else (0,0,0,0)

QUIET_LOGS = _b("QUIET_LOGS", "1")

# number of recent events
DASH_ROWS  = int(os.getenv("DASH_ROWS", "6"))
SHOW_PREVIEW = int(os.getenv("SHOW_PREVIEW", "0"))

# ---- camera / I/O ----
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
ROI          = _get_csv_ints("ROI", "0,0,0,0")      # x,y,w,h; 0s = full frame
SLEEP_SEC    = float(os.getenv("SLEEP_SEC", "1.0"))
SAVE_DIR     = os.getenv("SAVE_FRAME_DIR", "./frames")
DEBUG_DIR    = os.getenv("DEBUG_DUMP_DIR", "./debug")
DRAW_STATS   = _b("DRAW_LABEL_STATS","1")
WARP_SIZE    = (int(os.getenv("WARP_W","400")), int(os.getenv("WARP_H","560")))

# ---- SIM MODE ----
SIM              = int(os.getenv("SIM","0"))   # enable with -SIM 1
SIM_INTERVAL_SEC = float(os.getenv("SIM_INTERVAL_SEC","12"))
SIM_VALUES       = os.getenv("SIM_VALUES","3,0")  # sequence to alternate through (default 3,0)

# ---- WS grace/watchdog ----
ENFORCE_STARTUP_GRACE = int(os.getenv("ENFORCE_STARTUP_GRACE","1"))
STARTUP_GRACE_S       = float(os.getenv("STARTUP_GRACE_SEC","1800"))
ENFORCE_BOTH_WS       = int(os.getenv("ENFORCE_BOTH_WS","1"))
BOTH_WS_TIMEOUT_S     = float(os.getenv("BOTH_WS_TIMEOUT_SEC","600"))

SAVE_IMAGES     = int(os.getenv("SAVE_IMAGES", "0"))
SAVE_WARPS      = int(os.getenv("SAVE_WARPS", "0"))
SAVE_GRAYSCALE  = int(os.getenv("SAVE_GRAYSCALE", "0"))
SAVE_MASKS      = int(os.getenv("SAVE_MASKS", "0"))
DEBUG_EDGE_LOW  = int(os.getenv("DEBUG_EDGE_LOW","30"))
DEBUG_EDGE_HIGH = int(os.getenv("DEBUG_EDGE_HIGH","90"))
SAVE_RING_DEBUG = int(os.getenv("SAVE_RING_DEBUG","0"))

TEST_FILES        = os.getenv("TEST_FILES", "").strip()
TEST_INTERVAL_SEC = float(os.getenv("TEST_INTERVAL_SEC", "10"))
TEST_ENABLE       = int(os.getenv("TEST_ENABLE", "0"))  # 0 = camera, 1 = test images


LOG_REJECTIONS  = int(os.getenv("LOG_REJECTIONS","0"))

# ---- rim classifier knobs ----
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

# absolute pixel guards (optional/adaptive)
USE_ABS_PX_GUARDS = int(os.getenv("USE_ABS_PX_GUARDS","0")) # 0=off, 1=fixed, 2=adaptive
CARD_SHORT_MIN_PX = int(os.getenv("CARD_SHORT_MIN_PX", "100"))
CARD_SHORT_MAX_PX = int(os.getenv("CARD_SHORT_MAX_PX", "420"))
CARD_LONG_MIN_PX  = int(os.getenv("CARD_LONG_MIN_PX",  "160"))
CARD_LONG_MAX_PX  = int(os.getenv("CARD_LONG_MAX_PX",  "720"))

_ADAPT_SHORT = deque(maxlen=40)
_ADAPT_LONG  = deque(maxlen=40)
ADAPT_SLOP_S = 60
ADAPT_SLOP_L = 90

# ---- arming / reset policy ----
ARM_CONSEC_N       = int(os.getenv("ARM_CONSEC_N", "3"))
ARM_FACEUP_MIN     = int(os.getenv("ARM_FACEUP_MIN", "2"))
ZERO_UP_CONSEC_N   = int(os.getenv("ZERO_UP_CONSEC_N", "3"))

# send policy
RESEND_EVERY       = int(os.getenv("RESEND_EVERY", "5"))
USE_GRAYSCALE_ONLY = int(os.getenv("USE_GRAYSCALE_ONLY","1"))

# ---- websocket (DUAL DESTINATIONS) ----
from ws_mgr import WSManager
TABLET_WS_URL  = os.getenv("TABLET_WS_URL",  "ws://192.168.1.246:8765").strip()  # cards_detected
ARDUINO_WS_URL = os.getenv("ARDUINO_WS_URL", "ws://192.168.1.245:8888").strip()  # move_dealer_forward

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

def render_dashboard(face_up_now, cards_now, armed, arm_streak, zero_up_streak, peak_up, cur_codes):
    import os, sys
    from pyfiglet import Figlet

    # fast console clear
    sys.stdout.write("\033[2J\033[H")

    # colors
    RESET  = "\033[0m"; BOLD   = "\033[1m"
    GREEN  = "\033[1;32m"; RED = "\033[1;31m"; YELLOW="\033[1;33m"
    CYAN   = "\033[36m"; GRAY  = "\033[90m"; BLUE = "\033[34m"

    DASH_BIGTEXT = int(os.getenv("DASH_BIGTEXT","0"))
    dash_rows    = int(os.getenv("DASH_ROWS","5"))

    # --- Header ---
    print(f"{BOLD}\033[34m=== WED NIGHT POKER — CARD DETECTOR ==={RESET}")  # title blue
    t_ok = ws_tablet.is_connected; a_ok = ws_arduino.is_connected
    t_color = GREEN if t_ok else RED
    a_color = GREEN if a_ok else RED
    print(f"{BOLD}Tablet : {t_color}{TABLET_WS_URL}{RESET}   [{t_color}{'UP' if t_ok else 'DOWN'}{RESET}]")
    print(f"{BOLD}Arduino: {a_color}{ARDUINO_WS_URL}{RESET}   [{a_color}{'UP' if a_ok else 'DOWN'}{RESET}]")

    # --- Current recognized codes (pretty), up to 5 ---
    pretty_codes = "  ".join(cur_codes[:5]) if cur_codes else ""
    if DASH_BIGTEXT and pretty_codes:
        ascii_line = "".join(INV_SYM.get(ch, ch) for ch in pretty_codes)  # QS 10D 4C
        f2 = Figlet(font="big")
        for line in f2.renderText(ascii_line).rstrip().splitlines():
            print(GREEN + line + RESET)
        print(f"{BOLD}{YELLOW}{pretty_codes}{RESET}\n")
    elif pretty_codes:
        print(f"{BOLD}Codes:{RESET} {GREEN}{pretty_codes}{RESET}")

    # --- UP : DOWN summary ---
    down_now = max(0, cards_now - face_up_now)
    if DASH_BIGTEXT:
        f = Figlet(font="big")
        big_up_lines   = f.renderText(str(face_up_now)).rstrip().splitlines()
        big_down_lines = f.renderText(str(down_now)).rstrip().splitlines()
        colon_lines    = f.renderText(":").rstrip().splitlines()

        max_lines = max(len(big_up_lines), len(colon_lines), len(big_down_lines))
        big_up_lines   += [""] * (max_lines - len(big_up_lines))
        colon_lines    += [""] * (max_lines - len(colon_lines))
        big_down_lines += [""] * (max_lines - len(big_down_lines))

        print()
        for u, c, d in zip(big_up_lines, colon_lines, big_down_lines):
            print(f"{GREEN}{u:<15}{YELLOW}{c:<15}{RED}{d}{RESET}")
        print()
        print(f"{BOLD}{YELLOW}CARDS  {GREEN}UP{RESET}:{RED}DOWN{RESET}\n")
    else:
        print(f"{BOLD}{YELLOW}UP{RESET}:{GREEN}{face_up_now}{RESET}  "
              f"{YELLOW}DOWN{RESET}:{RED}{down_now}{RESET}\n")

    # --- State line ---
    if armed:
        state_txt = f"{GREEN}ARMED{RESET}"
    else:
        state_txt = f"{YELLOW}RESET{RESET}" if zero_up_streak >= ZERO_UP_CONSEC_N else f"{YELLOW}IDLE{RESET}"

    print(f"{BOLD}[LIVE]{RESET} Cards={YELLOW}{cards_now}{RESET}  "
          f"State={state_txt}  Arm={arm_streak}/{ARM_CONSEC_N}  "
          f"Zero={zero_up_streak}/{ZERO_UP_CONSEC_N}  Peak={peak_up}")

    # --- Events (optional) ---
    if dash_rows > 0 and events:
        print(f"{GRAY}" + "-" * 72 + f"{RESET}")
        print("Recent events:")
        for line in list(events)[:dash_rows]:
            color = GREEN if "ARMED" in line else (YELLOW if "RESET" in line else (RED if "fail" in line.lower() else GRAY))
            print("  " + color + line + RESET)

    sys.stdout.flush()

def _startup_summary():
    t_ok = ws_tablet.wait_connected(2.0)
    a_ok = ws_arduino.wait_connected(2.0)
    log_event(f"Tablet {'connected' if t_ok else 'pending'}")
    log_event(f"Arduino {'connected' if a_ok else 'pending'}")

# ---------------- SIM MODE LOOP ----------------
def run_simulator():
    try:
        seq = [int(s.strip()) for s in SIM_VALUES.split(",") if s.strip() != ""]
    except ValueError:
        seq = [3, 0]
    if not seq: seq = [3, 0]

    log_event(f"SIM MODE: sequence={seq} interval={SIM_INTERVAL_SEC:.1f}s  (to Tablet only)")
    idx = 0
    arm_streak = 0; zero_up_streak = 0; peak_up = 0
    while True:
        val = int(seq[idx % len(seq)])
        peak_up = max(peak_up, val)
        ws_tablet.send_cards_detected(val)  # no codes in SIM
        log_event(f"[SIM] cards_detected -> {val}")
        render_dashboard(face_up_now=val, cards_now=val, armed=False,
                         arm_streak=arm_streak, zero_up_streak=zero_up_streak,
                         peak_up=peak_up, cur_codes=[])
        time.sleep(SIM_INTERVAL_SEC)
        idx += 1

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

def _rej(msg):
    if LOG_REJECTIONS: print(msg)

def _card_candidates(mask, frame_w, frame_h):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 3000: _rej(f"[rej] area<3000 area={area:.0f}"); continue
        x,y,w,h = cv2.boundingRect(cnt)
        aabb_frac = (w*h)/float(frame_w*frame_h + 1e-6)
        if not (CARD_AABB_MIN_FRAC <= aabb_frac <= CARD_AABB_MAX_FRAC):
            _rej(f"[rej] aabb_frac={aabb_frac:.5f} wh=({w},{h})"); continue
        if min(w,h) < CARD_SHORTSIDE_MIN:
            _rej(f"[rej] shortside_min={min(w,h)} < {CARD_SHORTSIDE_MIN}"); continue

        hull = cv2.convexHull(cnt)
        solidity = area / (cv2.contourArea(hull)+1e-6)
        (cx,cy),(mw,mh),ang = cv2.minAreaRect(cnt)
        asp = max(mw,mh)/max(1.0,min(mw,mh))
        obb_fill = area / (mw*mh + 1e-6)
        if not (ASPECT_MIN <= asp <= ASPECT_MAX):
            _rej(f"[rej] aspect={asp:.3f} not in [{ASPECT_MIN},{ASPECT_MAX}]"); continue
        if solidity < SOLIDITY_MIN or obb_fill < FILL_OBB_MIN:
            _rej(f"[rej] solidity/fill={solidity:.3f}/{obb_fill:.3f}"); continue

        quad = cv2.boxPoints(cv2.minAreaRect(cnt))
        if not _right_angle_quad(quad):
            _rej("[rej] angles off 90±tol"); continue

        short_px = min(mw, mh); long_px = max(mw, mh)
        _ADAPT_SHORT.append(short_px); _ADAPT_LONG.append(long_px)

        if USE_ABS_PX_GUARDS == 1:
            if not (CARD_SHORT_MIN_PX <= short_px <= CARD_SHORT_MAX_PX and
                    CARD_LONG_MIN_PX  <= long_px  <= CARD_LONG_MAX_PX):
                _rej(f"[rej] abs_px short={short_px:.0f} long={long_px:.0f}"); continue
        elif USE_ABS_PX_GUARDS == 2 and len(_ADAPT_SHORT) >= 5:
            import numpy as _np
            ms, ml = _np.median(_ADAPT_SHORT), _np.median(_ADAPT_LONG)
            if not (ms-ADAPT_SLOP_S <= short_px <= ms+ADAPT_SLOP_S and
                    ml-ADAPT_SLOP_L <= long_px  <= ml+ADAPT_SLOP_L):
                _rej(f"[rej] abs_px_adapt short={short_px:.0f}/{ms:.0f} long={long_px:.0f}/{ml:.0f}")
                continue
        yield cnt

# -------------- ring-only classifier --------------
# ( #2 ) Precompute ring/center indices per (shape, params) to avoid recomputing masks every card
_RING_CACHE = {}
def _ring_indices(shape, rim_frac, inner_frac):
    key = (shape, rim_frac, inner_frac)
    hit = _RING_CACHE.get(key)
    if hit is not None: return hit
    h, w = shape
    rim   = int(max(1, rim_frac   * min(h, w)))
    inner = int(max(rim+1, inner_frac * min(h, w)))
    band = np.zeros((h, w), dtype=bool)
    band[:inner,:]  = True;  band[-inner:,:]  = True
    band[:, :inner] = True;  band[:, -inner:] = True
    inner_cut = np.zeros((h, w), dtype=bool)
    inner_cut[:rim,:]  = True; inner_cut[-rim:,:]  = True
    inner_cut[:, :rim] = True; inner_cut[:, -rim:] = True
    ring = band & (~inner_cut)
    csz = max(10, min(w, h)//5)
    cx0 = w//2 - csz//2; cy0 = h//2 - csz//2
    ctr_slice = (slice(cy0, cy0+csz), slice(cx0, cx0+csz))
    _RING_CACHE[key] = (ring, ctr_slice)
    return _RING_CACHE[key]

def classify_by_white_rim(warp_bgr, make_panel=True):
    lab = cv2.cvtColor(warp_bgr, cv2.COLOR_BGR2LAB)
    L   = lab[...,0].astype(np.float32)
    a   = lab[...,1].astype(np.float32) - 128.0
    b   = lab[...,2].astype(np.float32) - 128.0
    C   = np.hypot(a, b)

    ring, ctr = _ring_indices(L.shape, RIM_OUTER_FRAC, RIM_INNER_FRAC)

    L_ring = L[ring]
    if L_ring.size == 0:
        return "face_down", {"rule":"no_ring"}, None

    L_thr = np.percentile(L_ring, REL_WHITE_PCT * 100.0)
    white_mask = (L >= L_thr) & ring if USE_GRAYSCALE_ONLY else ((L >= L_thr) & (C <= CHROMA_MAX) & ring)
    white_frac = float(white_mask.mean())

    normL = _normalize_L(L.astype(np.uint8))
    edges = cv2.Canny(normL, DEBUG_EDGE_LOW, DEBUG_EDGE_HIGH)
    rim_edge = float((edges[ring] > 0).mean())
    center_edge = float((edges[ctr] > 0).mean())

    stats = {"crm": round(white_frac,3), "rim_e": round(rim_edge,3),
             "center_ed": round(center_edge,3), "L_thr": float(L_thr)}

    label = "face_up" if (white_frac >= FACEUP_WHITE_MIN and rim_edge <= RIM_EDGE_MAX and center_edge <= CENTER_EDGE_MAX) else "face_down"

    panel = None
    # ( #7 ) Avoid building heavy debug panels unless requested
    if (SAVE_WARPS or SAVE_RING_DEBUG) and make_panel:
        overlay = warp_bgr.copy()
        ring_vis = np.zeros_like(overlay)
        ring_vis[white_mask.astype(bool)] = (0,255,0)
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

def send_cards_change_or_every(count: int, codes=None):
    global _last_obs, _same_streak
    count = int(count)
    if _last_obs is None or count != _last_obs:
        _last_obs = count; _same_streak = 1
        ws_tablet.send_cards_detected(count, codes=codes)
        log_event(f"cards_detected -> {count} (changed) codes={codes or []}")
        return True
    _same_streak += 1
    if _same_streak >= RESEND_EVERY:
        _same_streak = 0
        ws_tablet.send_cards_detected(count, codes=codes)
        log_event(f"cards_detected -> {count} (periodic) codes={codes or []}")
        return True
    return False

def send_reset_reliably():
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

# === NEW: Background classification thread pool (simple, safe, no external deps) ===
BG_CLASSIFY       = int(os.getenv("BG_CLASSIFY", "1"))
CLASSIFY_WORKERS  = int(os.getenv("CLASSIFY_WORKERS", "5"))  # You asked for 5 (usually 3 active on flop)
CLASSIFY_Q_MAX    = int(os.getenv("CLASSIFY_Q_MAX", "20"))

class ClassifyWorkers:
    """
    Minimal thread pool:
      - submit(slot_key, warp_gray/bgr) from main thread
      - worker threads run _classify_card(warp)
      - results drained in main loop via fetch_all()
    Uses an 'inflight' set so the same slot isn't re-submitted every frame.
    """
    def __init__(self, n_workers=5, q_max=20):
        self.jobs = queue.Queue(maxsize=max(5, q_max))
        self.results = queue.Queue()
        self._stop = threading.Event()
        self.inflight = set()
        self._lock = threading.Lock()
        self.workers = []
        for _ in range(max(1, n_workers)):
            t = threading.Thread(target=self._run, daemon=True)
            t.start()
            self.workers.append(t)

    def _run(self):
        while not self._stop.is_set():
            try:
                epoch, slot_idx, warp = self.jobs.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                out = _classify_card(warp)  # returns (code, score[, rot])
                if len(out) == 3:
                    code, score, _rot = out
                else:
                    code, score = out
                self.results.put((epoch, slot_idx, (code, float(score or 0.0))))
            except Exception:
                # keep running; report empty result
                self.results.put((epoch, slot_idx, (None, 0.0)))
            finally:
                with self._lock:
                    self.inflight.discard((epoch, slot_idx))
                self.jobs.task_done()

    def submit(self, epoch, slot_idx, warp):
        with self._lock:
            key = (epoch, slot_idx)
            if key in self.inflight:
                return False
            try:
                self.jobs.put_nowait((epoch, slot_idx, warp))
                self.inflight.add(key)
                return True
            except queue.Full:
                return False

    def fetch_all(self):
        out = []
        while True:
            try:
                out.append(self.results.get_nowait())
            except queue.Empty:
                break
        return out

    def stop(self):
        self._stop.set()

# === NEW: Stable 5-slot tracker (flop, flop, flop, turn, river) ===
# The key property: slots never reorder. We match by SHA-1 digest first;
# if not found, we optionally match by proximity; otherwise we fill the first empty slot.
class SlotTracker:
    def __init__(self, max_slots=5, prox_gate_px=80):
        self.max_slots = max_slots
        self.prox_gate = int(prox_gate_px)
        self.slots = [None] * self.max_slots  # list of dicts or None
        self.epoch = 0  # increments on RESET to invalidate in-flight jobs

    def clear(self):
        self.slots = [None] * self.max_slots
        self.epoch += 1  # invalidate any in-flight jobs

    def _nearest_slot(self, x, y):
        best_i, best_d = None, 1e9
        for i, s in enumerate(self.slots):
            if not s: continue
            dx = abs(s["x"] - x); dy = abs(s["y"] - y)
            d = (dx*dx + dy*dy) ** 0.5
            if d < best_d:
                best_i, best_d = i, d
        if best_i is not None and best_d <= self.prox_gate:
            return best_i
        return None

    def faceup_count(self):
        return sum(1 for s in self.slots if s is not None)

    def codes_list(self):
        # Only return codes for occupied slots, in fixed slot order
        return [s["code"] for s in self.slots if s is not None]

    def codes_list_fixed5(self):
        # Always length 5, using "UNK" for empty
        out = []
        for s in self.slots:
            out.append(s["code"] if s else "UNK")
        return out

    def update_from_detections(self, detections, frame_idx):
        """
        detections: list of dicts with keys {x, y, digest, warp}
        Returns: (changed: bool) if slot assignment or content changed
        """
        changed = False

        # First pass: exact digest matches update in place
        unmatched = []
        for det in detections:
            matched = False
            for i, s in enumerate(self.slots):
                if s is None: continue
                if s["digest"] == det["digest"]:
                    # Update position/warp/last_seen
                    s["x"], s["y"] = det["x"], det["y"]
                    s["warp"] = det["warp"]
                    s["last_seen"] = frame_idx
                    matched = True
                    break
            if not matched:
                unmatched.append(det)

        # Second pass: proximity match to existing slots
        still_unmatched = []
        for det in unmatched:
            i = self._nearest_slot(det["x"], det["y"])
            if i is not None:
                s = self.slots[i]
                # Only adopt if this slot wasn't updated this frame (avoid stealing)
                # We check last_seen < frame_idx to see if not updated by digest pass.
                if s["last_seen"] < frame_idx:
                    s["x"], s["y"] = det["x"], det["y"]
                    s["warp"] = det["warp"]
                    s["last_seen"] = frame_idx
                else:
                    still_unmatched.append(det)
            else:
                still_unmatched.append(det)

        # Third pass: assign any remaining detections to the first empty slots
        for det in still_unmatched:
            for i in range(self.max_slots):
                if self.slots[i] is None:
                    self.slots[i] = {
                        "digest": det["digest"],
                        "x": det["x"], "y": det["y"],
                        "warp": det["warp"],
                        "code": "UNK",
                        "score": 0.0,
                        "last_seen": frame_idx,
                    }
                    changed = True
                    break
            # If all 5 already occupied, we ignore extras

        # Garbage-collect any slots that disappeared (optional): we *do not* auto-clear
        # here to keep stability per your requirements. Reset logic will clear all.

        return changed

    def improve_code(self, slot_idx, code, score):
        s = self.slots[slot_idx]
        if s is None: return False
        prev = s["code"]
        if (prev == "UNK" and code) or (score is not None and score > (s["score"] or 0.0)):
            s["code"] = code
            s["score"] = float(score or 0.0)
            return True
        return False



class FrameSource:
    """
    Provides frames from either a real camera or a list of images.
    - Camera mode: .read() -> (ok, frame) from cv2.VideoCapture
    - Test mode: cycles through provided images; advances to next image every TEST_INTERVAL_SEC
                 (holds the same frame between advances)
    """
    def __init__(self, camera_index, test_files_csv, interval_sec):
        self.use_files = bool(test_files_csv)
        self.interval = float(interval_sec)
        self._last_switch = time.time()
        self._idx = 0
        self._imgs = []
        self.cap = None

        if self.use_files:
            paths = [p.strip() for p in test_files_csv.split(",") if p.strip()]
            for p in paths:
                img = cv2.imread(p)
                if img is None:
                    print(f"[TEST] WARNING: could not read image: {p}")
                else:
                    self._imgs.append(img)
            if not self._imgs:
                raise RuntimeError("[TEST] No readable images for TEST_FILES")
            print(f"[TEST] Loaded {len(self._imgs)} test frame(s); cycling every {self.interval:.1f}s")
        else:
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                raise RuntimeError("[ERROR] camera not found / cannot open")
            try:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            except Exception:
                pass

    def read(self):
        if not self.use_files:
            return self.cap.read()

        now = time.time()
        # advance to next file every interval
        if now - self._last_switch >= self.interval:
            self._idx = (self._idx + 1) % len(self._imgs)
            self._last_switch = now
        frame = self._imgs[self._idx]
        return True, frame.copy()

    def release(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass




# -------------- main --------------

# ( #6 ) move constants out of hot loop
CLASSIFY_THRESH = 0.55
DASH_EVERY_N = max(1, int(os.getenv("DASH_EVERY_N", "5")))   # ( #3 ) throttle dashboard redraws

def _sha1_digest_of_warp(warp_bgr):
    """ === NEW: Stable, consistent digest of the warp for identity & change detection === """
    g = cv2.cvtColor(warp_bgr, cv2.COLOR_BGR2GRAY)
    # Using the full bytes is fine; SHA-1 cost is negligible vs classification
    h = hashlib.sha1(g.tobytes()).hexdigest()
    # Shorten for compactness; 10 hex chars (40 bits) is plenty for 5 cards
    return h[:10]

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)

    SAVE_WARP_RAW = int(os.getenv("SAVE_WARP_RAW", "0"))
    if SAVE_WARP_RAW:
        os.makedirs(os.path.join(DEBUG_DIR, "warps_raw"), exist_ok=True)

    _startup_summary()
    # templates presence warning
    try:
        full_dir = os.getenv("CARD_FULL_TEMPL_DIR", "./templates")
        files = [f for f in os.listdir(full_dir) if f.lower().endswith((".png",".jpg",".jpeg"))]
        if len(files) < 52:
            log_event(f"WARNING: expected 52 full-card templates in {full_dir} (found {len(files)})")
    except FileNotFoundError:
        log_event("WARNING: full-card template dir not found — place 52 images in ./templates")

    log_event("loop starting")

    if SIM:
        run_simulator()
        return


    # Unified frame source (camera OR test images)
    try:
        src = FrameSource(CAMERA_INDEX,
                      TEST_FILES if TEST_ENABLE else "",
                      TEST_INTERVAL_SEC)

    except Exception as e:
        print(str(e))
        return

    app_start_ts = time.time()
    ws_watchdog_start = app_start_ts
    both_connected_once = False

    armed = False
    arm_streak = 0
    zero_up_streak = 0
    peak_up = 0

    # detection-order codes now derived from slots; kept for dashboard uses
    det_codes = []  # e.g. ["7H","JC","3D","UNK","UNK"]

    # for periodic sender sync when we push a codes update
    global _last_obs, _same_streak
    _last_obs = None
    _same_streak = 0

    frame_idx = 0  # ( #3 ) dashboard throttle counter

    # === NEW: Slot tracker & background workers ===
    SLOT_PROX_GATE_PX = int(os.getenv("SLOT_PROX_GATE_PX", "80"))
    CLASSIFY_WORKERS = int(os.getenv("CLASSIFY_WORKERS", "5"))

    slots = SlotTracker(max_slots=5, prox_gate_px=SLOT_PROX_GATE_PX)
    workers = ClassifyWorkers(n_workers=CLASSIFY_WORKERS, q_max=CLASSIFY_Q_MAX) if BG_CLASSIFY else None


    while True:
        ts = time.strftime("%Y%m%d-%H%M%S")
        ok, frame = src.read()


        # ---- Both-WS policy (startup grace + runtime watchdog) ----
        now = time.time()
        both_up = (ws_tablet.is_connected and ws_arduino.is_connected)
        if not both_connected_once:
            if both_up:
                both_connected_once = True
                ws_watchdog_start = now
                log_event("Both WS connected — runtime watchdog ARMED")
            else:
                if ENFORCE_STARTUP_GRACE and (now - app_start_ts >= STARTUP_GRACE_S):
                    log_event(f"EXIT: startup grace ({int(STARTUP_GRACE_S)}s) expired without both WS up")
                    if frame_idx % DASH_EVERY_N == 0:
                        render_dashboard(0,0,False,0,0,0, cur_codes=[])
                    sys.exit(3)
        else:
            if both_up:
                ws_watchdog_start = now
            else:
                if ENFORCE_BOTH_WS and (now - ws_watchdog_start >= BOTH_WS_TIMEOUT_S):
                    log_event(f"EXIT: both WS not up simultaneously for {int(now - ws_watchdog_start)}s")
                    if frame_idx % DASH_EVERY_N == 0:
                        render_dashboard(0,0,False,0,0,0, cur_codes=[])
                    sys.exit(2)

        if not ok:
            time.sleep(SLEEP_SEC)
            if frame_idx % DASH_EVERY_N == 0:
                render_dashboard(0, 0, armed, arm_streak, zero_up_streak, peak_up, cur_codes=[])
            frame_idx += 1
            continue

        # ---- preprocess & threshold ----
        proc, offx, offy = _roi(frame, ROI) if sum(ROI) != 0 else (frame, 0, 0)
        gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        norm = _normalize_L(gray)
        _, th = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # ( #7 ) only compute/save extras when requested
        if SAVE_MASKS:
            edges_full = cv2.Canny(norm, DEBUG_EDGE_LOW, DEBUG_EDGE_HIGH)
            cv2.imwrite(os.path.join(SAVE_DIR, f"norm_{ts}.png"), norm)
            cv2.imwrite(os.path.join(SAVE_DIR, f"th_{ts}.png"), th)
            cv2.imwrite(os.path.join(SAVE_DIR, f"edges_{ts}.png"), edges_full)

        # ---- detect cards (warp + rim-classify) ----
        cards = []
        for idx, cnt in enumerate(_card_candidates(th, proc.shape[1], proc.shape[0])):
            warp, box, (mw, mh) = _warp_from_contour(proc, cnt)
            label, info, panel = classify_by_white_rim(warp)
            cards.append((label, info, box, panel, warp))

        # ---- build detections for SLOT tracker (only face-up cards) ----
        detections = []
        for (label, _info, box, _panel, warp) in cards:
            if label != "face_up":
                continue
            xs = box[:,0]; ys = box[:,1]
            x_center = float(xs.min() + xs.max()) * 0.5
            y_center = float(ys.min() + ys.max()) * 0.5
            digest = _sha1_digest_of_warp(warp)
            detections.append({"x": x_center, "y": y_center, "digest": digest, "warp": warp})

        prev_count = slots.faceup_count()
        # Update slot assignments; never reorders, just fills first empty slots
        changed_slots = slots.update_from_detections(detections, frame_idx)
        face_up_now = slots.faceup_count()
        cards_now   = len(cards)
        peak_up     = max(peak_up, face_up_now)

        # Build codes (only for occupied slots, in fixed order)
        det_codes = slots.codes_list()

        def _codes_for_tablet():
            if face_up_now < ARM_FACEUP_MIN or face_up_now == 0:
                return None
            # send only as many codes as face_up_now (consumer expects length == count)
            return list(det_codes)

        # >>> EARLY NOTIFY: send immediately if face-up count changed <<<
        if face_up_now > ARM_FACEUP_MIN and face_up_now != prev_count:
            send_cards_change_or_every(face_up_now, codes=_codes_for_tablet())
        # <<< END EARLY NOTIFY >>>

        # ---- state machine: send COUNT immediately ----
        if not armed:
            arm_streak = arm_streak + 1 if face_up_now >= ARM_FACEUP_MIN else 0
            if arm_streak >= ARM_CONSEC_N:
                armed = True
                send_cards_change_or_every(face_up_now, codes=_codes_for_tablet())
                zero_up_streak = 0
                peak_up = face_up_now
                log_event(f"STATE -> ARMED (need {ARM_FACEUP_MIN}+ for {ARM_CONSEC_N})")
        else:
            zero_up_streak = zero_up_streak + 1 if face_up_now == 0 else 0
            if zero_up_streak >= ZERO_UP_CONSEC_N:
                log_event(f"STATE -> RESET (zero-up for {ZERO_UP_CONSEC_N}, peak={peak_up})")
                send_cards_change_or_every(0, codes=[])  # explicit zero
                send_reset_reliably()
                armed = False
                arm_streak = 0
                zero_up_streak = 0
                peak_up = 0
                det_codes = []
                # === NEW: clear slots and bump epoch to invalidate any in-flight classify jobs
                slots.clear()
            else:
                send_cards_change_or_every(face_up_now, codes=_codes_for_tablet())

        # ---- background classification for UNKNOWN slots ----
        improved = False
        if face_up_now >= ARM_FACEUP_MIN:
            # submit jobs for UNK codes
            for i, s in enumerate(slots.slots):
                if s is None: continue
                if s["code"] != "UNK": continue
                warp = s.get("warp")
                if warp is None: continue
                if BG_CLASSIFY and workers:
                    workers.submit(slots.epoch, i, warp)
                else:
                    # synchronous fallback
                    out = _classify_card(warp)
                    if len(out) == 3: code, score, _rot = out
                    else:             code, score = out
                    if code and score >= CLASSIFY_THRESH:
                        if slots.improve_code(i, code.upper(), score):
                            improved = True

            # drain background results (non-blocking)
            if BG_CLASSIFY and workers:
                for (epoch, idx, (code, score)) in workers.fetch_all():
                    # Ignore stale results from previous epochs (after RESET)
                    if epoch != slots.epoch:
                        continue
                    if code and score >= CLASSIFY_THRESH:
                        if slots.improve_code(idx, code.upper(), score):
                            improved = True

        # Pull latest codes (occupied slots only, in slot order)
        det_codes = slots.codes_list()

        if improved and face_up_now >= ARM_FACEUP_MIN:
            ws_tablet.send_cards_detected(face_up_now, codes=list(det_codes))
            log_event(f"codes_update -> {face_up_now}")
            _last_obs = face_up_now
            _same_streak = 0

        # ---- pretty codes for dashboard (detection order) ----
        cur_codes_pretty = det_codes[:]

        # ( #3 ) throttle dashboard redraws
        if frame_idx % DASH_EVERY_N == 0:
            render_dashboard(face_up_now, cards_now, armed, arm_streak, zero_up_streak, peak_up, cur_codes_pretty)

        # ---- optional live preview ----
        if SHOW_PREVIEW:
            cv2.imshow("Card Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_idx += 1
        time.sleep(SLEEP_SEC)

    # cleanup
    if workers: workers.stop()
    src.release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

