#!/usr/bin/env python3
# Wednesday Night Poker — Card Detector
# v2025.10.24 — single-WS (Hub) refactor + clear logging
# Simplified: card_detect only sends detected cards; dealer advancement logic moved to tablet

import os, sys, cv2, time, atexit, numpy as np, logging, hashlib, threading, queue
from collections import deque
from dotenv import load_dotenv
from pyfiglet import Figlet
from rank_suit import classify_fullcard_anyrot as _classify_card  # returns (code, score, rot)

# ---- OpenCV perf knobs ----
try:
    cv2.setUseOptimized(True)
    cv2.setNumThreads(0)
except Exception:
    pass

# ---------------- env ----------------
if os.path.exists("env"): load_dotenv("env")
def _b(k, d="0"): return os.getenv(k, d).strip().lower() in ("1","true","yes")

def parse_dash_args(argv):
    args = {}; key=None
    for token in argv:
        if token.startswith('-') and not token.startswith('--'):
            key = token.lstrip('-'); args[key]=None
        else:
            if key: args[key]=token; key=None
    return args

_cli = parse_dash_args(sys.argv[1:])
if _cli: print("[CLI overrides detected]", _cli)
for k, v in _cli.items(): os.environ[k] = v

def _get_csv_ints(name, default="0,0,0,0"):
    raw = os.getenv(name, default)
    raw = raw.split("#", 1)[0].strip().strip('"').strip("'")
    parts = [p.strip() for p in raw.split(",")]
    return tuple(int(p or "0") for p in parts[:4]) if len(parts) >= 4 else (0,0,0,0)

QUIET_LOGS = _b("QUIET_LOGS", "1")

DASH_ROWS     = int(os.getenv("DASH_ROWS", "6"))
SHOW_PREVIEW  = int(os.getenv("SHOW_PREVIEW", "0"))

# ---- camera / I/O ----
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
ROI          = _get_csv_ints("ROI", "0,0,0,0")
SLEEP_SEC    = float(os.getenv("SLEEP_SEC", "1.0"))
SAVE_DIR     = os.getenv("SAVE_FRAME_DIR", "./frames")
DEBUG_DIR    = os.getenv("DEBUG_DUMP_DIR", "./debug")
DRAW_STATS   = _b("DRAW_LABEL_STATS","1")
WARP_SIZE    = (int(os.getenv("WARP_W","400")), int(os.getenv("WARP_H","560")))

# ---- SIM MODE ----
SIM              = int(os.getenv("SIM","0"))
SIM_INTERVAL_SEC = float(os.getenv("SIM_INTERVAL_SEC","12"))
SIM_VALUES       = os.getenv("SIM_VALUES","3,0")

# ---- WS grace/watchdog (unused now, but left for easy toggling if desired) ----
ENFORCE_STARTUP_GRACE = int(os.getenv("ENFORCE_STARTUP_GRACE","0"))
STARTUP_GRACE_S       = float(os.getenv("STARTUP_GRACE_SEC","1800"))
ENFORCE_BOTH_WS       = int(os.getenv("ENFORCE_BOTH_WS","0"))
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
TEST_ENABLE       = int(os.getenv("TEST_ENABLE", "0"))

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

USE_ABS_PX_GUARDS = int(os.getenv("USE_ABS_PX_GUARDS","0"))
CARD_SHORT_MIN_PX = int(os.getenv("CARD_SHORT_MIN_PX", "100"))
CARD_SHORT_MAX_PX = int(os.getenv("CARD_SHORT_MAX_PX", "420"))
CARD_LONG_MIN_PX  = int(os.getenv("CARD_LONG_MIN_PX",  "160"))
CARD_LONG_MAX_PX  = int(os.getenv("CARD_LONG_MAX_PX",  "720"))

_ADAPT_SHORT = deque(maxlen=40)
_ADAPT_LONG  = deque(maxlen=40)
ADAPT_SLOP_S = 60
ADAPT_SLOP_L = 90

# ---- slot forget/TTL ----
SLOT_FORGET_FRAMES = int(os.getenv("SLOT_FORGET_FRAMES", "8"))

# send policy
RESEND_EVERY       = int(os.getenv("RESEND_EVERY", "5"))
USE_GRAYSCALE_ONLY = int(os.getenv("USE_GRAYSCALE_ONLY","1"))

# ---------------- dashboard helpers (make sure events exists before log_event!) ----------------
events = deque(maxlen=DASH_ROWS)
hub_client_count = 0  # Cached client count from hub
hub_client_count_ts = 0  # Timestamp of last fetch

def log_event(s: str):
    ts = time.strftime("%H:%M:%S"); events.appendleft(f"{ts}  {s}")

# ---------------- WebSocket: single connection to the Hub ----------------
from poker_hub_client import WSManager
HUB_WS_URL = os.getenv("HUB_WS_URL", "ws://192.168.1.54:8888").strip()

hub_client_info = None  # Cached client info from hub
hub_client_info_ts = 0  # Timestamp of last fetch

def _fetch_hub_client_count():
    """Fetch client count and info from hub HTTP status endpoint. Returns (count, clients_list) or (None, None) on error."""
    global hub_client_count, hub_client_count_ts, hub_client_info, hub_client_info_ts
    try:
        import urllib.request
        import json
        # Extract host:port from WS URL (e.g., ws://192.168.1.54:8888 -> http://192.168.1.54:8787)
        ws_url = HUB_WS_URL.replace("ws://", "http://").replace("wss://", "https://")
        # Use HTTP bridge port (default 8787)
        http_port = int(os.getenv("HTTP_BRIDGE_PORT", "8787"))
        if ":" in ws_url:
            host = ws_url.split("://")[1].split(":")[0]
            status_url = f"http://{host}:{http_port}/status"
        else:
            status_url = f"http://{ws_url.split('://')[1]}:{http_port}/status"
        
        with urllib.request.urlopen(status_url, timeout=1.0) as response:
            data = json.loads(response.read().decode())
            count = data.get("total_clients", 0)
            clients_list = data.get("clients", [])
            hub_client_count = count
            hub_client_info = clients_list
            hub_client_count_ts = time.time()
            hub_client_info_ts = time.time()
            return count, clients_list
    except Exception:
        # On error, return cached value if recent (within 10 seconds)
        if time.time() - hub_client_info_ts < 10:
            return hub_client_count, hub_client_info
        return None, None

# Get client name from env or default to "detector"
CLIENT_NAME = os.getenv("CLIENT_NAME", "detector").strip()
ws_hub = WSManager(url=HUB_WS_URL, on_event=log_event, name="Hub", role="detector", client_name=CLIENT_NAME)

def _attach_hub_send_logger():
    import json as _json
    _orig = ws_hub.send_json
    def _wrapped(payload: dict) -> bool:
        ok = _orig(payload)
        cmd = payload.get("command") if isinstance(payload, dict) else None
        try:
            body = _json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        except Exception:
            body = str(payload)
        if len(body) > 200:
            body = body[:200] + "…"
        log_event(f"[WS→HUB] cmd={cmd or '?'} ok={ok} connected={ws_hub.is_connected} payload={body}")
        return ok
    ws_hub.send_json = _wrapped

_attach_hub_send_logger()
ws_hub.start()
atexit.register(ws_hub.stop)

RESET  = "\033[0m"; BOLD="\033[1m"; GREEN="\033[1;32m"; RED="\033[1;31m"; YELLOW="\033[1;33m"; CYAN="\033[36m"; GRAY="\033[90m"; BLUE="\033[34m"

# ---- logging quiet mode ----
if QUIET_LOGS:
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("websocket").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

def render_dashboard(face_up_now, cards_now, cur_codes):
    import sys
    # Clear screen and move cursor to top-left for in-place updates
    # Use escape sequences to clear screen and reset cursor position
    if sys.stdout.isatty():
        # Flush first to ensure previous output is written
        sys.stdout.flush()
        # Clear entire screen (\033[2J) and move cursor to home (\033[H)
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()
    else:
        # If not a TTY, just print a separator line
        print("\n" + "=" * 72 + "\n")

    DASH_BIGTEXT = int(os.getenv("DASH_BIGTEXT","0"))
    dash_rows    = int(os.getenv("DASH_ROWS","5"))

    print(f"{BOLD}\033[34m=== WED NIGHT POKER — CARD DETECTOR (via HUB) ==={RESET}")

    h_ok = ws_hub.is_connected
    h_color = GREEN if h_ok else RED
    print(f"{BOLD}Hub    : {h_color}{HUB_WS_URL}{RESET}   [{h_color}{'UP' if h_ok else 'DOWN'}{RESET}]")
    
    # Fetch and display client count and names
    client_count, clients_list = _fetch_hub_client_count()
    if client_count is not None:
        count_color = GREEN if client_count > 0 else GRAY
        print(f"{BOLD}Clients : {count_color}{client_count}{RESET}", end="")
        # Show client names if available
        if clients_list:
            names = []
            for client in clients_list:
                name = client.get("name")
                role = client.get("role", "unknown")
                # Use name if available, otherwise use role
                if name:
                    # Rename "detector" name to "Card Detector" for display
                    display_name = "Card Detector" if name.lower() == "detector" else name
                else:
                    # Rename "detector" role to "Card Detector" for display
                    display_name = "Card Detector" if role == "detector" else role
                names.append(display_name)
            if names:
                print(f"  {GRAY}[{', '.join(names)}]{RESET}")
            else:
                print()
        else:
            print()
    else:
        print(f"{BOLD}Clients : {GRAY}?{RESET}")

    pretty_codes = "  ".join(cur_codes[:5]) if cur_codes else ""
    if DASH_BIGTEXT and pretty_codes:
        ascii_line = "".join({'S':'S','H':'H','D':'D','C':'C','1':'1','0':'0'}.get(ch, ch) for ch in pretty_codes)
        f2 = Figlet(font="big")
        for line in f2.renderText(ascii_line).rstrip().splitlines():
            print(GREEN + line + RESET)
        print(f"{BOLD}{YELLOW}{pretty_codes}{RESET}\n")
    elif pretty_codes:
        print(f"{BOLD}Codes:{RESET} {GREEN}{pretty_codes}{RESET}")

    down_now = max(0, cards_now - face_up_now)
    print(f"{BOLD}{YELLOW}UP{RESET}:{GREEN}{face_up_now}{RESET}  "
          f"{YELLOW}DOWN{RESET}:{RED}{down_now}{RESET}\n")

    print(f"{BOLD}[LIVE]{RESET} Cards={YELLOW}{cards_now}{RESET}  "
          f"Face Up={GREEN}{face_up_now}{RESET}")

    if dash_rows > 0 and events:
        print(f"{GRAY}" + "-" * 72 + f"{RESET}")
        print("Recent events:")
        for line in list(events)[:dash_rows]:
            color = GREEN if "detected" in line.lower() else (RED if "fail" in line.lower() else GRAY)
            print("  " + color + line + RESET)
    sys.stdout.flush()

def _startup_summary():
    h_ok = ws_hub.wait_connected(2.0)
    log_event(f"Hub {'connected' if h_ok else 'pending'}")

# ---------------- SIM MODE LOOP ----------------
def _send_cards_to_hub(count, codes=None):
    payload = {"command": "cards_detected", "data": {"count": int(count)}}
    if codes is not None:
        payload["data"]["codes"] = [str(c).upper() for c in (codes or [])]
    ok = ws_hub.send_json(payload)
    log_event(f"[WS→HUB] cards_detected count={count} codes={codes or []} ok={ok}")
    return ok

def run_simulator():
    try:
        seq = [int(s.strip()) for s in os.getenv("SIM_VALUES","3,0").split(",") if s.strip()!=""]
    except:
        seq = [3, 0]
    if not seq: seq=[3,0]
    idx=0; peak_up=0
    log_event(f"SIM MODE: sequence={seq} interval={SIM_INTERVAL_SEC:.1f}s  (to HUB)")
    while True:
        val = int(seq[idx % len(seq)])
        _send_cards_to_hub(val)
        log_event(f"[SIM] cards_detected -> {val}")
        render_dashboard(face_up_now=val, cards_now=val, cur_codes=[])
        time.sleep(SIM_INTERVAL_SEC); idx+=1

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

def _right_angle_quad(quad, tol):
    p = np.array(quad, dtype=np.float32)
    v = np.roll(p, -1, axis=0) - p
    for i in range(4):
        a = v[i-1]; b = v[i]
        c = np.degrees(np.arccos(np.clip(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-6), -1, 1)))
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
    if W > H: ow, oh = oh, ow
    src = _order_quad(box)
    dst = np.array([[0,0],[ow-1,0],[ow-1,oh-1],[0,oh-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warp = cv2.warpPerspective(img_bgr, M, (ow, oh))
    return warp, box, (W,H)


# -------------- ring-only classifier helpers --------------
_RING_CACHE = {}
def _ring_indices(shape, rim_frac, inner_frac):
    """
    Returns (ring_mask, center_slice) for fast "white rim" detection.
    - ring_mask: boolean array selecting a border ring around the card
    - center_slice: tuple of slices selecting a central patch (for edge check)
    """
    key = (shape, rim_frac, inner_frac)
    hit = _RING_CACHE.get(key)
    if hit is not None:
        return hit

    h, w = shape
    # rim thickness and inner boundary expressed as fractions of the short side
    rim   = int(max(1, rim_frac    * min(h, w)))
    inner = int(max(rim + 1, inner_frac * min(h, w)))

    # Build a rectangular "band" of thickness `inner` on all four sides…
    band = np.zeros((h, w), dtype=bool)
    band[:inner, :]  = True
    band[-inner:, :] = True
    band[:, :inner]  = True
    band[:, -inner:] = True

    # …and carve out the inner-most part to leave a ring of thickness `rim`
    inner_cut = np.zeros((h, w), dtype=bool)
    inner_cut[:rim, :]  = True
    inner_cut[-rim:, :] = True
    inner_cut[:, :rim]  = True
    inner_cut[:, -rim:] = True

    ring = band & (~inner_cut)

    # A small central square for checking interior edges (should be quiet for face-up)
    csz = max(10, min(w, h) // 5)
    cx0 = w // 2 - csz // 2
    cy0 = h // 2 - csz // 2
    ctr_slice = (slice(cy0, cy0 + csz), slice(cx0, cx0 + csz))

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
    if USE_GRAYSCALE_ONLY:
        white_mask = (L >= L_thr) & ring
    else:
        white_mask = (L >= L_thr) & (C <= CHROMA_MAX) & ring

    white_frac = float(white_mask.mean())

    normL = _normalize_L(L.astype(np.uint8))
    edges = cv2.Canny(normL, DEBUG_EDGE_LOW, DEBUG_EDGE_HIGH)
    rim_edge = float((edges[ring] > 0).mean())
    center_edge = float((edges[ctr] > 0).mean())

    stats = {
        "crm": round(white_frac,3),
        "rim_e": round(rim_edge,3),
        "center_ed": round(center_edge,3),
        "L_thr": float(L_thr)
    }

    label = "face_up" if (
        white_frac >= FACEUP_WHITE_MIN and
        rim_edge   <= RIM_EDGE_MAX    and
        center_edge<= CENTER_EDGE_MAX
    ) else "face_down"

    panel = None
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
            _rej(f"[rej] aspect={asp:.3f}"); continue
        if solidity < SOLIDITY_MIN or obb_fill < FILL_OBB_MIN:
            _rej(f"[rej] solidity/fill={solidity:.3f}/{obb_fill:.3f}"); continue
        quad = cv2.boxPoints(cv2.minAreaRect(cnt))
        if not _right_angle_quad(quad, ANGLE_TOL_DEG):
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
                _rej(f"[rej] abs_px_adapt short={short_px:.0f}/{ms:.0f} long={long_px:.0f}/{ml:.0f}"); continue
        yield cnt

# -------------- WS helpers --------------
_last_obs = None
_same_streak = 0

def send_cards_change_or_every(count: int, codes=None):
    global _last_obs, _same_streak
    count = int(count)
    if _last_obs is None or count != _last_obs:
        _last_obs = count; _same_streak = 1
        _send_cards_to_hub(count, codes=codes)
        return True
    _same_streak += 1
    if _same_streak >= RESEND_EVERY:
        _same_streak = 0
        _send_cards_to_hub(count, codes=codes)
        return True
    return False

# === Background classification ===
BG_CLASSIFY       = int(os.getenv("BG_CLASSIFY", "1"))
CLASSIFY_WORKERS  = int(os.getenv("CLASSIFY_WORKERS", "5"))
CLASSIFY_Q_MAX    = int(os.getenv("CLASSIFY_Q_MAX", "20"))

class ClassifyWorkers:
    def __init__(self, n_workers=5, q_max=20):
        self.jobs = queue.Queue(maxsize=max(5, q_max))
        self.results = queue.Queue()
        self._stop = threading.Event()
        self.inflight = set()
        self._lock = threading.Lock()
        self.workers = []
        for _ in range(max(1, n_workers)):
            t = threading.Thread(target=self._run, daemon=True)
            t.start(); self.workers.append(t)
    def _run(self):
        while not self._stop.is_set():
            try: epoch, slot_idx, warp = self.jobs.get(timeout=0.25)
            except queue.Empty: continue
            try:
                out = _classify_card(warp)
                if len(out) == 3: code, score, _rot = out
                else:             code, score = out
                self.results.put((epoch, slot_idx, (code, float(score or 0.0))))
            except Exception as e:
                # Log errors in background worker
                import traceback
                print(f"[BG Worker ERROR] slot {slot_idx}: {e}")
                if "QUIET" not in str(os.getenv("QUIET_LOGS", "")):
                    traceback.print_exc()
                self.results.put((epoch, slot_idx, (None, 0.0)))
            finally:
                with self._lock: self.inflight.discard((epoch, slot_idx))
                self.jobs.task_done()
    def submit(self, epoch, slot_idx, warp):
        with self._lock:
            key = (epoch, slot_idx)
            if key in self.inflight: return False
            try:
                self.jobs.put_nowait((epoch, slot_idx, warp))
                self.inflight.add(key); return True
            except queue.Full:
                return False
    def fetch_all(self):
        out=[]
        while True:
            try: out.append(self.results.get_nowait())
            except queue.Empty: break
        return out
    def stop(self): self._stop.set()

# === Stable 5-slot tracker with GC ===
class SlotTracker:
    def __init__(self, max_slots=5, prox_gate_px=80):
        self.max_slots = max_slots
        self.prox_gate = int(prox_gate_px)
        self.slots = [None] * self.max_slots
        self.epoch = 0
    def clear(self):
        self.slots = [None] * self.max_slots
        self.epoch += 1
    def _nearest_slot(self, x, y):
        best_i, best_d = None, 1e9
        for i, s in enumerate(self.slots):
            if not s: continue
            dx = abs(s["x"] - x); dy = abs(s["y"] - y)
            d = (dx*dx + dy*dy) ** 0.5
            if d < best_d: best_i, best_d = i, d
        return best_i if (best_i is not None and best_d <= self.prox_gate) else None
    def faceup_count(self):
        return sum(1 for s in self.slots if s is not None)
    def codes_list(self):
        return [s["code"] for s in self.slots if s is not None]
    def codes_list_fixed5(self):
        return [s["code"] if s else "UNK" for s in self.slots]
    def update_from_detections(self, detections, frame_idx):
        changed = False
        unmatched = []
        for det in detections:
            matched = False
            for i, s in enumerate(self.slots):
                if s is None: continue
                if s["digest"] == det["digest"]:
                    s["x"], s["y"] = det["x"], det["y"]
                    s["warp"] = det["warp"]
                    s["last_seen"] = frame_idx
                    matched = True
                    break
            if not matched: unmatched.append(det)
        still_unmatched = []
        for det in unmatched:
            i = self._nearest_slot(det["x"], det["y"])
            if i is not None:
                s = self.slots[i]
                if s["last_seen"] < frame_idx:
                    s["x"], s["y"] = det["x"], det["y"]
                    s["warp"] = det["warp"]
                    s["last_seen"] = frame_idx
                else:
                    still_unmatched.append(det)
            else:
                still_unmatched.append(det)
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
        return changed
    def gc_expired(self, frame_idx, forget_frames=SLOT_FORGET_FRAMES):
        changed = False
        cutoff = frame_idx - max(1, int(forget_frames))
        for i, s in enumerate(self.slots):
            if s is None: continue
            if s["last_seen"] < cutoff:
                self.slots[i] = None
                changed = True
        return changed
    def improve_code(self, slot_idx, code, score):
        s = self.slots[slot_idx]
        if s is None: return False
        prev = s["code"]
        prev_score = s.get("score", 0.0) or 0.0
        
        # Only update if:
        # 1. Previous was UNK and we have a code, OR
        # 2. New score is better - use smaller margin for low-confidence matches
        if prev == "UNK":
            score_margin = 0.0  # Always accept first match
        elif prev_score < 0.60:
            score_margin = 0.05  # Low confidence - allow small improvements
        else:
            score_margin = 0.15  # High confidence - require big improvement
        
        if (prev == "UNK" and code) or (score is not None and score > prev_score + score_margin):
            s["code"] = code
            s["score"] = float(score or 0.0)
            return True
        return False

class FrameSource:
    def __init__(self, camera_index, test_files_csv, interval_sec):
        self.use_files = bool(test_files_csv)
        self.interval = float(interval_sec)
        self._last_switch = time.time()
        self._idx = 0
        self._imgs = []; self.cap = None
        if self.use_files:
            paths = [p.strip() for p in test_files_csv.split(",") if p.strip()]
            for p in paths:
                img = cv2.imread(p)
                if img is None: print(f"[TEST] WARNING: could not read image: {p}")
                else: self._imgs.append(img)
            if not self._imgs: raise RuntimeError("[TEST] No readable images for TEST_FILES")
            print(f"[TEST] Loaded {len(self._imgs)} test frame(s); cycling every {self.interval:.1f}s")
        else:
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened(): raise RuntimeError("[ERROR] camera not found / cannot open")
            try:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            except Exception: pass
    def read(self):
        if not self.use_files: return self.cap.read()
        now = time.time()
        if now - self._last_switch >= self.interval:
            self._idx = (self._idx + 1) % len(self._imgs)
            self._last_switch = now
        frame = self._imgs[self._idx]
        return True, frame.copy()
    def release(self):
        if self.cap is not None:
            try: self.cap.release()
            except Exception: pass

# -------------- main --------------
CLASSIFY_THRESH = float(os.getenv("CLASSIFY_THRESH", os.getenv("FULLCARD_MIN_SCORE", "0.55")))
DASH_EVERY_N = max(1, int(os.getenv("DASH_EVERY_N", "5")))

def _sha1_digest_of_warp(warp_bgr):
    g = cv2.cvtColor(warp_bgr, cv2.COLOR_BGR2GRAY)
    h = hashlib.sha1(g.tobytes()).hexdigest()
    return h[:10]

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)
    SAVE_WARP_RAW = int(os.getenv("SAVE_WARP_RAW", "0"))
    if SAVE_WARP_RAW:
        os.makedirs(os.path.join(DEBUG_DIR, "warps_raw"), exist_ok=True)

    _startup_summary()
    try:
        full_dir = os.getenv("CARD_FULL_TEMPL_DIR", "./templates")
        files = [f for f in os.listdir(full_dir) if f.lower().endswith((".png",".jpg",".jpeg"))]
        if len(files) < 52: log_event(f"WARNING: expected 52 full-card templates in {full_dir} (found {len(files)})")
    except FileNotFoundError:
        log_event("WARNING: full-card template dir not found — place 52 images in ./templates")
    log_event("loop starting")

    if SIM: return run_simulator()

    try:
        src = FrameSource(CAMERA_INDEX, TEST_FILES if TEST_ENABLE else "", TEST_INTERVAL_SEC)
    except Exception as e:
        print(str(e)); return

    init_card_detect_sent = False

    global _last_obs, _same_streak
    _last_obs = None; _same_streak = 0

    frame_idx = 0
    SLOT_PROX_GATE_PX = int(os.getenv("SLOT_PROX_GATE_PX", "80"))
    slots   = SlotTracker(max_slots=5, prox_gate_px=SLOT_PROX_GATE_PX)
    workers = ClassifyWorkers(n_workers=CLASSIFY_WORKERS, q_max=CLASSIFY_Q_MAX) if BG_CLASSIFY else None

    while True:
        ok, frame = src.read()
        now = time.time()

        # Light status note (no hard exit)
        if not ws_hub.is_connected and frame_idx % max(1, DASH_EVERY_N*2) == 0:
            log_event("Hub not connected yet…")

        if not ok:
            time.sleep(SLEEP_SEC)
            if frame_idx % DASH_EVERY_N == 0:
                render_dashboard(0, 0, cur_codes=[])
            frame_idx += 1
            continue

        if not init_card_detect_sent:
            _send_cards_to_hub(0, codes=[])
            init_card_detect_sent = True
            log_event("init card detect sent")

        # ---- preprocess ----
        proc, offx, offy = _roi(frame, ROI) if sum(ROI) != 0 else (frame, 0, 0)
        gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        norm = _normalize_L(gray)
        _, th = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if SAVE_MASKS:
            edges_full = cv2.Canny(norm, DEBUG_EDGE_LOW, DEBUG_EDGE_HIGH)
            ts = time.strftime("%Y%m%d-%H%M%S")
            cv2.imwrite(os.path.join(SAVE_DIR, f"norm_{ts}.png"), norm)
            cv2.imwrite(os.path.join(SAVE_DIR, f"th_{ts}.png"), th)
            cv2.imwrite(os.path.join(SAVE_DIR, f"edges_{ts}.png"), edges_full)

        # ---- detect cards ----
        cards = []
        for cnt in _card_candidates(th, proc.shape[1], proc.shape[0]):
            warp, box, (mw, mh) = _warp_from_contour(proc, cnt)
            label, info, panel = classify_by_white_rim(warp)
            cards.append((label, info, box, panel, warp))

        # ---- build detections for SLOT tracker (only face-up) ----
        detections = []
        for (label, _info, box, _panel, warp) in cards:
            if label != "face_up": continue
            xs = box[:,0]; ys = box[:,1]
            x_center = float(xs.min() + xs.max()) * 0.5
            y_center = float(ys.min() + ys.max()) * 0.5
            digest = _sha1_digest_of_warp(warp)
            detections.append({"x": x_center, "y": y_center, "digest": digest, "warp": warp})

        prev_count = slots.faceup_count()

        # Update slots with current detections + GC so counts can drop
        changed_slots = slots.update_from_detections(detections, frame_idx)
        if slots.gc_expired(frame_idx, SLOT_FORGET_FRAMES):
            changed_slots = True

        face_up_now = slots.faceup_count()
        cards_now   = len(cards)
        det_codes   = slots.codes_list()

        # Send card updates whenever the count changes
        if face_up_now != prev_count:
            codes_to_send = list(det_codes) if face_up_now > 0 else []
            send_cards_change_or_every(face_up_now, codes=codes_to_send)

        # ---- background classification for UNKNOWN slots ----
        improved = False
        if face_up_now > 0:
            unk_count = sum(1 for s in slots.slots if s and s.get("code") == "UNK")
            for i, s in enumerate(slots.slots):
                if s is None: continue
                # Classify UNK slots, or re-classify low-confidence matches (might be wrong)
                current_score = s.get("score", 0.0) or 0.0
                if s["code"] != "UNK" and current_score >= 0.65:
                    continue  # Skip high-confidence matches (don't waste time re-classifying)
                warp = s.get("warp")
                if warp is None: continue
                if BG_CLASSIFY and workers:
                    workers.submit(slots.epoch, i, warp)
                else:
                    try:
                        out = _classify_card(warp)
                        if len(out) == 3: code, score, _rot = out
                        else:             code, score = out
                        # Log classification results (throttled to avoid spam)
                        if code and score >= CLASSIFY_THRESH:
                            log_event(f"✓ Classified slot {i}: {code} (score={score:.3f})")
                        elif frame_idx % 30 == 0:  # Log failures less frequently
                            log_event(f"✗ Slot {i}: {code or 'None'} (score={score:.3f}, need {CLASSIFY_THRESH})")
                        if code and score >= CLASSIFY_THRESH:
                            if slots.improve_code(i, code.upper(), score):
                                improved = True
                    except Exception as e:
                        log_event(f"Classification error slot {i}: {e}")
                        if not QUIET_LOGS:
                            import traceback
                            traceback.print_exc()
                        continue
            if BG_CLASSIFY and workers:
                all_results = workers.fetch_all()
                if all_results and frame_idx % 20 == 0:
                    log_event(f"BG workers returned {len(all_results)} results")
                for (epoch, idx, (code, score)) in all_results:
                    if epoch != slots.epoch:
                        if frame_idx % 30 == 0:
                            log_event(f"BG result epoch mismatch: {epoch} vs {slots.epoch} for slot {idx}")
                        continue
                    # Log classification results from background workers
                    if code and score >= CLASSIFY_THRESH:
                        log_event(f"✓ Classified slot {idx}: {code} (score={score:.3f}) [BG]")
                        if slots.improve_code(idx, code.upper(), score):
                            improved = True
                    elif frame_idx % 20 == 0:  # Log failures more frequently
                        log_event(f"✗ Slot {idx}: {code or 'None'} (score={score:.3f}, need {CLASSIFY_THRESH}) [BG]")

        det_codes = slots.codes_list()
        if improved and face_up_now > 0:
            _send_cards_to_hub(face_up_now, codes=list(det_codes))
            log_event(f"codes_update -> {face_up_now}")
            _last_obs = face_up_now; _same_streak = 0

        if frame_idx % DASH_EVERY_N == 0:
            render_dashboard(face_up_now, cards_now, cur_codes=det_codes)

        if SHOW_PREVIEW:
            cv2.imshow("Card Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        frame_idx += 1
        time.sleep(SLEEP_SEC)

    # cleanup
    if workers: workers.stop()
    src.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

