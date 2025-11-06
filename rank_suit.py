# rank_suit.py
import os, cv2, numpy as np
from functools import lru_cache

# Accept both "10" and "T"
_RMAP = {"A":"A","J":"J","Q":"Q","K":"K",
         "T":"10","10":"10","2":"2","3":"3","4":"4","5":"5","6":"6","7":"7","8":"8","9":"9"}
_SUITS = ("S","H","D","C")


# --- rank-corner ROI (same coordinates for both warp and templates) ---
def _rank_roi_rect(h, w):
    # small, safe ROI around the rank character in the top-left corner
    m = min(h, w)
    pad = int(0.02 * m)
    box = int(0.16 * m)             # you can try 0.18-0.20 if your corner rank is larger
    y0, x0 = pad, pad
    y1, x1 = y0 + box, x0 + int(0.75 * box)
    return (y0, y1, x0, x1)

# --- suit-center ROI for distinguishing hearts vs diamonds ---
def _suit_center_roi_rect(h, w):
    # ROI in the center of the card where suit symbols are prominent
    m = min(h, w)
    center_y, center_x = h // 2, w // 2
    roi_size = int(0.25 * m)  # 25% of the short side
    y0 = max(0, center_y - roi_size // 2)
    y1 = min(h, center_y + roi_size // 2)
    x0 = max(0, center_x - roi_size // 2)
    x1 = min(w, center_x + roi_size // 2)
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
    bank = _load_templates()
    if not bank:
        return None, 0.0

    H, W = bank[0][1].shape[:2]
    rank_rect = _rank_roi_rect(H, W)

    # 1) full-image sweep (0° and 180° like you had)
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

    # 2) tie-break for close matches, especially when distinguishing similar cards
    (s1, c1, t1, g1), (s2, c2, t2, g2) = top2
    R1, S1 = c1[:-1], c1[-1]
    R2, S2 = c2[:-1], c2[-1]

    close = (s1 - s2) <= 0.02             # margin you can tune
    same_rank = (R1 == R2)
    different_suit = (S1 != S2)
    same_suit = (S1 == S2)
    small_ranks = {R1, R2}.issubset({"2","3","4","5"})  # Include rank 5
    medium_ranks = {R1, R2}.issubset({"4","5","6","7"})  # Ranks that can be confused
    red_suit_confusion = {S1, S2} == {"H", "D"}  # Hearts vs Diamonds confusion

    # Case 1: Same rank, different suit (especially H vs D) - use suit center ROI
    if close and same_rank and different_suit and red_suit_confusion:
        suit_rect = _suit_center_roi_rect(H, W)
        gs1 = _roi(g1, suit_rect); ts1 = _roi(t1, suit_rect)
        gs2 = _roi(g2, suit_rect); ts2 = _roi(t2, suit_rect)
        s1s = cv2.matchTemplate(gs1, ts1, cv2.TM_CCOEFF_NORMED)[0][0]
        s2s = cv2.matchTemplate(gs2, ts2, cv2.TM_CCOEFF_NORMED)[0][0]
        if s2s > s1s + 1e-4:  # tiny epsilon
            return c2, float(s2)
        # else fall through to c1
    
    # Case 2: Same rank, different suit (general case) - use suit center ROI
    elif close and same_rank and different_suit:
        suit_rect = _suit_center_roi_rect(H, W)
        gs1 = _roi(g1, suit_rect); ts1 = _roi(t1, suit_rect)
        gs2 = _roi(g2, suit_rect); ts2 = _roi(t2, suit_rect)
        s1s = cv2.matchTemplate(gs1, ts1, cv2.TM_CCOEFF_NORMED)[0][0]
        s2s = cv2.matchTemplate(gs2, ts2, cv2.TM_CCOEFF_NORMED)[0][0]
        if s2s > s1s + 1e-4:
            return c2, float(s2)
    
    # Case 3: Same suit, different rank (e.g., 5H vs 6H) - use rank corner ROI
    elif close and same_suit and not same_rank and medium_ranks:
        # rank-only correlation in the corner ROI (higher resolution, more discriminative)
        gr1 = _roi(g1, rank_rect); tr1 = _roi(t1, rank_rect)
        gr2 = _roi(g2, rank_rect); tr2 = _roi(t2, rank_rect)
        s1r = cv2.matchTemplate(gr1, tr1, cv2.TM_CCOEFF_NORMED)[0][0]
        s2r = cv2.matchTemplate(gr2, tr2, cv2.TM_CCOEFF_NORMED)[0][0]
        if s2r > s1r + 1e-4:  # tiny epsilon
            return c2, float(s2)
        # else fall through to c1
    
    # Case 4: Original logic - same suit, small ranks - use rank corner ROI
    elif close and same_suit and small_ranks:
        # rank-only correlation in the corner ROI (higher resolution, more discriminative)
        gr1 = _roi(g1, rank_rect); tr1 = _roi(t1, rank_rect)
        gr2 = _roi(g2, rank_rect); tr2 = _roi(t2, rank_rect)
        s1r = cv2.matchTemplate(gr1, tr1, cv2.TM_CCOEFF_NORMED)[0][0]
        s2r = cv2.matchTemplate(gr2, tr2, cv2.TM_CCOEFF_NORMED)[0][0]
        if s2r > s1r + 1e-4:  # tiny epsilon
            return c2, float(s2)
        # else fall through to c1
    
    return c1, float(s1)


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

