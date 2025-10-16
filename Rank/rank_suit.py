# rank_suit.py
import os, cv2, numpy as np
from functools import lru_cache

# Accept both "10" and "T"
_RMAP = {"A":"A","J":"J","Q":"Q","K":"K",
         "T":"10","10":"10","2":"2","3":"3","4":"4","5":"5","6":"6","7":"7","8":"8","9":"9"}
_SUITS = ("S","H","D","C")

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
    best_code, best_score = None, -1.0

    # Try 0° and 180° (many cams flip the card)
    candidates = [gwarp, cv2.rotate(gwarp, cv2.ROTATE_180)]
    for g in candidates:
        for code, templ in bank:
            # cv2.TM_CCOEFF_NORMED expects both same size
            score = cv2.matchTemplate(g, templ, cv2.TM_CCOEFF_NORMED)[0][0]
            if score > best_score:
                best_score, best_code = score, code
    return best_code, float(best_score)

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

