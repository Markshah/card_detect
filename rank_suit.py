# rank_suit.py
import os, cv2, numpy as np
import math
from functools import lru_cache

# Accept both "10" and "T"
_RMAP = {"A":"A","J":"J","Q":"Q","K":"K",
         "T":"10","10":"10","2":"2","3":"3","4":"4","5":"5","6":"6","7":"7","8":"8","9":"9"}
_SUITS = ("S","H","D","C")


USE_PIP_TIEBREAK   = int(os.getenv("USE_PIP_TIEBREAK", "1"))
PIP_TIEBREAK_MARGIN = float(os.getenv("PIP_TIEBREAK_MARGIN", "0.025"))  # NCC closeness


# --- rank-corner ROI (same coordinates for both warp and templates) ---
def _rank_roi_rect(h, w):
    # small, safe ROI around the rank character in the top-left corner
    m = min(h, w)
    pad = int(0.02 * m)
    box = int(0.16 * m)             # you can try 0.18-0.20 if your corner rank is larger
    y0, x0 = pad, pad
    y1, x1 = y0 + box, x0 + int(0.75 * box)
    return (y0, y1, x0, x1)

def _roi(img, rect):
    y0, y1, x0, x1 = rect
    return img[y0:y1, x0:x1]


def _code_ok(name):
    base = os.path.splitext(name)[0].upper()
    # Expect rank then suit. e.g., QS, 10D
    if len(base) < 2: return False
    suit = base[-1]
    rank = base[:-1]
    rank = _RMAP.get(rank, None)
    return suit in _SUITS and rank is not None

def _code_from_name(name):
    base = os.path.splitext(name)[0].upper()
    suit = base[-1]
    rank = _RMAP.get(base[:-1], base[:-1])
    return f"{rank}{suit}"

@lru_cache(maxsize=1)
def _load_templates():
    """Returns list of (code, gray_template) normalized to same size."""
    templ_dir = os.getenv("CARD_FULL_TEMPL_DIR", "./templates")
    files = [f for f in os.listdir(templ_dir) if f.lower().endswith((".png",".jpg",".jpeg")) and _code_ok(f)]
    if not files:
        return []

    # Load all and normalize to median size to reduce scale sensitivity
    imgs = []
    for f in files:
        p = os.path.join(templ_dir, f)
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        imgs.append((f, img))

    # Choose target size
    hws = np.array([(im.shape[0], im.shape[1]) for _, im in imgs])
    med_h, med_w = np.median(hws, axis=0).astype(int)
    med_h = max(200, int(med_h))  # keep reasonable size
    med_w = max(140, int(med_w))

    bank = []
    for f, im in imgs:
        imr = cv2.resize(im, (med_w, med_h), interpolation=cv2.INTER_AREA)
        imr = cv2.GaussianBlur(imr, (3,3), 0)  # mild blur for robustness
        bank.append((_code_from_name(f), imr))
    return bank  # [(code, gray)]

def _prep(gray):
    g = gray
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (_load_templates()[0][1].shape[1], _load_templates()[0][1].shape[0]),
                   interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(g, (3,3), 0)
    return g


def _best_match(gwarp):
    import os
    bank = _load_templates()
    if not bank:
        return None, 0.0

    H, W = bank[0][1].shape[:2]
    rank_rect = _rank_roi_rect(H, W)

    # 1) full-image sweep (0° and 180°)
    top2 = []  # list of (score, code, templ_gray, candidate_gray)
    candidates = [gwarp, cv2.rotate(gwarp, cv2.ROTATE_180)]
    for g in candidates:
        for code, templ in bank:
            s = cv2.matchTemplate(g, templ, cv2.TM_CCOEFF_NORMED)[0][0]
            if len(top2) < 2:
                top2.append((s, code, templ, g))
                top2.sort(reverse=True, key=lambda x: x[0])
            else:
                if s > top2[-1][0]:
                    top2[-1] = (s, code, templ, g)
                    top2.sort(reverse=True, key=lambda x: x[0])

    if not top2:
        return None, 0.0
    if len(top2) == 1:
        return top2[0][1], float(top2[0][0])

    # 2) existing tie-break (corner ROI) only for 2/3/4 when scores are close
    (s1, c1, t1, g1), (s2, c2, t2, g2) = top2
    R1, S1 = c1[:-1], c1[-1]
    R2, S2 = c2[:-1], c2[-1]

    close = (s1 - s2) <= 0.02            # your original margin
    small_ranks = {R1, R2}.issubset({"2","3","4"})
    same_suit = (S1 == S2)

    if close and small_ranks and same_suit:
        gr1 = _roi(g1, rank_rect); tr1 = _roi(t1, rank_rect)
        gr2 = _roi(g2, rank_rect); tr2 = _roi(t2, rank_rect)
        s1r = cv2.matchTemplate(gr1, tr1, cv2.TM_CCOEFF_NORMED)[0][0]
        s2r = cv2.matchTemplate(gr2, tr2, cv2.TM_CCOEFF_NORMED)[0][0]
        if s2r > s1r + 1e-4:
            return c2, float(s2)
        # else fall through to c1

    # 3) NEW: pip-count tie-break (number cards only), gated by env
    try:
        USE_PIP_TIEBREAK = int(os.getenv("USE_PIP_TIEBREAK", "0"))
        PIP_TIEBREAK_MARGIN = float(os.getenv("PIP_TIEBREAK_MARGIN", "0.025"))
    except Exception:
        USE_PIP_TIEBREAK, PIP_TIEBREAK_MARGIN = 0, 0.025

    if USE_PIP_TIEBREAK and abs(s1 - s2) <= PIP_TIEBREAK_MARGIN:
        def _rank_to_int(r: str):
            if r in {"J","Q","K","A"}: return None
            r = "10" if r in {"T","10"} else r
            try: return int(r)
            except: return None

        r1 = _rank_to_int(R1)
        r2 = _rank_to_int(R2)

        if (r1 is not None) or (r2 is not None):
            # Count pips on the higher-score rotation image (g1). It's grayscale already.
            try:
                count = pip_count_center(g1)  # function added elsewhere
            except NameError:
                count = None  # pip counter not present; skip
            if count is not None:
                m1 = (r1 == count) if r1 is not None else False
                m2 = (r2 == count) if r2 is not None else False
                # If exactly one candidate rank matches the pip count, prefer it.
                if m1 ^ m2:
                    return (c1, float(s1)) if m1 else (c2, float(s2))

    # default: keep the top match
    return c1, float(s1)



# ---------- center-only pip counter (no corners) ----------
def pip_count_center(gray,
                     center_crop_frac: float = 0.12,
                     area_min: int = 350,
                     area_max: int = 15000,
                     close_k: int = 3,
                     open_k: int = 2,
                     dilate_k: int = 2,
                     circ_min: float = 0.35,
                     circ_max: float = 0.88,
                     min_center_gap_frac: float = 0.55) -> int | None:
    """
    Count suit pips using only the center of a warped card (portrait ~400x560).
    Returns 2..10 for number cards; None for face/unknown/ambiguous.
    """
    import cv2, numpy as np

    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    H, W = gray.shape
    x0, x1 = int(W * center_crop_frac), int(W * (1 - center_crop_frac))
    y0, y1 = int(H * center_crop_frac), int(H * (1 - center_crop_frac))
    roi = gray[y0:y1, x0:x1]

    roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)
    _, binv = cv2.threshold(roi_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    mask = cv2.morphologyEx(binv, cv2.MORPH_CLOSE, k_close, iterations=1)
    mask = cv2.morphologyEx(mask,  cv2.MORPH_OPEN,  k_open,  iterations=1)
    if dilate_k > 0:
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k)), iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    roi_h, roi_w = mask.shape
    for c in cnts:
        a = cv2.contourArea(c)
        if a < area_min or a > area_max:
            continue
        x, y, w, h = cv2.boundingRect(c)
        # reject blobs touching the crop border (likely inner frame/art)
        if x <= 0 or y <= 0 or (x + w) >= roi_w - 1 or (y + h) >= roi_h - 1:
            continue
        ar = w / float(h) if h else 0.0
        if not (0.5 <= ar <= 1.8):
            continue
        hull = cv2.convexHull(c)
        solidity = a / (cv2.contourArea(hull) + 1e-6)
        if solidity < 0.75:
            continue
        p = cv2.arcLength(c, True)
        circ = (4.0 * math.pi * a) / (p * p + 1e-6)
        if not (circ_min <= circ <= circ_max):
            continue
        M = cv2.moments(c)
        cx = (M["m10"] / (M["m00"] + 1e-6))
        cy = (M["m01"] / (M["m00"] + 1e-6))
        candidates.append({"c": c, "area": a, "cx": cx, "cy": cy, "scale": min(w, h)})

    if not candidates:
        return None

    # de-dup: merge split blobs by proximity
    candidates.sort(key=lambda d: d["area"], reverse=True)
    taken = []
    for d in candidates:
        close = False
        for k in taken:
            gap = min(d["scale"], k["scale"]) * min_center_gap_frac
            if ( (d["cx"]-k["cx"])**2 + (d["cy"]-k["cy"])**2 ) ** 0.5 < gap:
                close = True
                break
        if not close:
            taken.append(d)

    n = len(taken)
    total_ink = sum(d["area"] for d in taken) / float(roi_w * roi_h)

    # final sanity for number cards
    if 2 <= n <= 10 and 0.015 <= total_ink <= 0.25:
        return n
    return None


def classify_fullcard(warp_bgr, score_thresh=0.60):
    """
    Returns (code, score). code like 'QS', '10D', or None if below threshold.
    """
    bank = _load_templates()
    if not bank:
        return None, 0.0

    g = _prep(warp_bgr)
    code, score = _best_match(g)
    if score < score_thresh:
        return None, score
    return code, score


def _rotate(img, deg):
    if deg == 0:   return img
    if deg == 90:  return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180: return cv2.rotate(img, cv2.ROTATE_180)
    if deg == 270: return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img

def classify_fullcard_anyrot(warp_bgr, rotations=(0,90,180,270)):
    """
    Try classify_fullcard on 0/90/180/270° and return the best.
    Returns: (code, score, rotation_deg)  e.g., ('QH', 0.83, 90) or (None, 0.0, None)
    """
    best = (None, 0.0, None)
    for r in rotations:
        img = _rotate(warp_bgr, r)
        code, score = classify_fullcard(img)  # uses your existing template matcher
        if code and score > best[1]:
            best = (code, float(score), r)
    return best

