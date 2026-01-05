# rank_suit.py
import os, cv2, numpy as np
from functools import lru_cache
import re

# Accept both "10" and "T"
_RMAP = {"A":"A","J":"J","Q":"Q","K":"K",
         "T":"10","10":"10","2":"2","3":"3","4":"4","5":"5","6":"6","7":"7","8":"8","9":"9"}
_SUITS = ("S","H","D","C")

# Multi-template support: patterns like "AS.png", "AS_1.png", "AS_center.png", "AS_left.png"
# All variants of the same card code are grouped together
USE_MULTI_TEMPLATES = int(os.getenv("USE_MULTI_TEMPLATES", "1"))
USE_HIST_EQ = int(os.getenv("USE_HIST_EQ", "1"))  # Histogram equalization for lighting
USE_MULTI_SCALE = int(os.getenv("USE_MULTI_SCALE", "1"))  # Multi-scale matching
MULTI_SCALE_FACTORS = [0.95, 1.0, 1.05]  # Try 95%, 100%, 105% scale


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
    # Support patterns like "AS", "AS1", "AS2", "AS_1", "AS_center", etc.
    # Extract base code (before number suffix or underscore)
    # Match: AS, AS1, AS2, AS_1, AS_center, etc.
    base_match = re.match(r"^([A-Z0-9]+[SHDC])(?:\d+|_[A-Z0-9]+)?$", base)
    if not base_match:
        return False
    base_code = base_match.group(1)
    if len(base_code) < 2: return False
    suit = base_code[-1]
    rank = base_code[:-1]
    rank = _RMAP.get(rank, None)
    return suit in _SUITS and rank is not None

def _code_from_name(name):
    """Extract base card code from filename, ignoring number/variant suffixes."""
    base = os.path.splitext(name)[0].upper()
    # Match base code (e.g., "AS" from "AS1.png", "AS2.png", "AS_1.png", etc.)
    # Pattern: card code followed by optional digits or underscore suffix
    base_match = re.match(r"^([A-Z0-9]+[SHDC])(?:\d+|_[A-Z0-9]+)?$", base)
    if base_match:
        base_code = base_match.group(1)
    else:
        # Fallback: try to extract from start (before any digits)
        # Remove trailing digits
        base_code = re.sub(r"\d+$", "", base)
        if len(base_code) < 2:
            # Last resort: try to extract from end
            if len(base) >= 2:
                base_code = base[:2] if len(base) == 2 else base[-2:]
            else:
                return None
    
    if len(base_code) < 2:
        return None
    suit = base_code[-1]
    rank = _RMAP.get(base_code[:-1], base_code[:-1])
    return f"{rank}{suit}"

@lru_cache(maxsize=1)
def _load_templates():
    """
    Returns list of (code, gray_template) normalized to same size.
    Supports multiple templates per card (e.g., AS.png, AS_1.png, AS_center.png).
    All variants of the same card are included.
    """
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
        code = _code_from_name(f)
        if code:
            bank.append((code, imr))
    
    # If multi-template support is enabled, we keep all variants
    # Otherwise, we could dedupe by code, but keeping all is better for matching
    return bank  # [(code, gray)] - may have multiple entries per code

def _prep(gray, target_size=None):
    """Preprocess image with improved normalization for lighting variations."""
    g = gray
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    
    # Get target size from templates if not provided
    if target_size is None:
        bank = _load_templates()
        if not bank:
            return g
        target_size = (bank[0][1].shape[1], bank[0][1].shape[0])
    
    g = cv2.resize(g, target_size, interpolation=cv2.INTER_AREA)
    
    # Improved normalization for lighting variations
    # 1. Percentile-based normalization (more robust to outliers)
    lo, hi = np.percentile(g, (2, 98))
    if hi > lo:
        g = np.clip((g - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    
    # 2. Histogram equalization (optional, helps with varying lighting)
    if USE_HIST_EQ:
        g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(g)
    
    # 3. Mild blur for robustness
    g = cv2.GaussianBlur(g, (3,3), 0)
    
    return g

def _best_match_fast(gwarp, bank):
    """Fast matching - one template per card initially, then use all templates for top candidates."""
    if not bank:
        return None, 0.0
    
    H, W = bank[0][1].shape[:2]
    rank_rect = _rank_roi_rect(H, W)
    
    # Step 1: Fast initial matching with one template per card
    if USE_MULTI_TEMPLATES and len(bank) > 52:
        # Group by card code and use first template of each
        from collections import OrderedDict
        unique_cards = OrderedDict()
        for code, templ in bank:
            if code not in unique_cards:
                unique_cards[code] = templ
        initial_bank = list(unique_cards.items())
    else:
        initial_bank = bank
    
    # Get top 3 candidates from initial matching
    top3 = []  # list of (score, code)
    
    for code, templ in initial_bank:
        s = cv2.matchTemplate(gwarp, templ, cv2.TM_CCOEFF_NORMED)[0][0]
        
        if len(top3) < 3:
            top3.append((s, code))
            top3.sort(reverse=True, key=lambda x: x[0])
        else:
            if s > top3[-1][0]:
                top3[-1] = (s, code)
                top3.sort(reverse=True, key=lambda x: x[0])
    
    if not top3:
        return None, 0.0
    
    # Step 2: For top candidates, try all templates of those cards (better accuracy)
    top_codes = [code for _, code in top3[:3]]  # Top 3 candidates
    
    # Also include cards with similar ranks to top candidates (e.g., if 6H is top, also check 5H and 7H)
    def rank_value(r):
        if r == 'A': return 1
        if r == 'K': return 13
        if r == 'Q': return 12
        if r == 'J': return 11
        if r == '10': return 10
        try: return int(r)
        except: return 0
    
    def get_rank(code):
        return code[:-1] if len(code) > 1 else code[0]
    
    # Add similar-rank cards to refinement list
    for score, code in top3[:2]:  # Check top 2 for similar ranks
        rank = get_rank(code)
        suit = code[-1] if len(code) > 0 else ''
        rank_val = rank_value(rank)
        
        # Add adjacent ranks (rank-1 and rank+1) of same suit
        for adj_rank_val in [rank_val - 1, rank_val + 1]:
            if 2 <= adj_rank_val <= 9:  # Valid numeric rank
                adj_code = f"{adj_rank_val}{suit}"
                if adj_code not in top_codes:
                    top_codes.append(adj_code)
    
    # Get best match for each candidate using all their templates
    candidate_scores = {}  # {code: (best_score, best_templ)}
    
    for code in top_codes:
        best_score_for_code = -1.0
        best_templ_for_code = None
        for c, templ in bank:
            if c == code:
                s = cv2.matchTemplate(gwarp, templ, cv2.TM_CCOEFF_NORMED)[0][0]
                if s > best_score_for_code:
                    best_score_for_code = s
                    best_templ_for_code = templ
        if best_templ_for_code is not None:
            candidate_scores[code] = (best_score_for_code, best_templ_for_code)
    
    if not candidate_scores:
        # Fallback to initial match
        return top3[0][1], float(top3[0][0])
    
    # Sort by score
    sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1][0], reverse=True)
    best_code, (best_score, best_templ) = sorted_candidates[0]
    
    # Step 3: Aggressive tie-breaking for similar ranks (5 vs 6, 8 vs 9, etc.)
    if len(sorted_candidates) > 1:
        second_code, (second_score, second_templ) = sorted_candidates[1]
        
        # Check if ranks are similar (both numeric and close)
        def rank_value(r):
            if r == 'A': return 1
            if r == 'K': return 13
            if r == 'Q': return 12
            if r == 'J': return 11
            if r == '10': return 10
            try: return int(r)
            except: return 0
        
        best_rank = best_code[:-1] if len(best_code) > 1 else best_code[0]
        second_rank = second_code[:-1] if len(second_code) > 1 else second_code[0]
        
        rank_diff = abs(rank_value(best_rank) - rank_value(second_rank))
        same_suit = best_code[-1] == second_code[-1] if len(best_code) > 0 and len(second_code) > 0 else False
        
        # If ranks are close (diff <= 1) and same suit, use rank ROI tie-breaking
        if rank_diff <= 1 and same_suit and abs(best_score - second_score) < 0.10:
            gr1 = _roi(gwarp, rank_rect)
            tr1 = _roi(best_templ, rank_rect)
            tr2 = _roi(second_templ, rank_rect)
            s1r = cv2.matchTemplate(gr1, tr1, cv2.TM_CCOEFF_NORMED)[0][0]
            s2r = cv2.matchTemplate(gr1, tr2, cv2.TM_CCOEFF_NORMED)[0][0]
            # Use rank ROI as tie-breaker - it's more discriminative for similar ranks
            if s2r > s1r + 0.01:  # Slightly larger margin for rank ROI
                return second_code, float(second_score)
    
    return best_code, float(best_score)

def _best_match(gwarp):
    bank = _load_templates()
    if not bank:
        return None, 0.0

    H, W = bank[0][1].shape[:2]
    rank_rect = _rank_roi_rect(H, W)

    # 1) full-image sweep (0° and 180° rotations)
    # With multi-scale and multi-template support
    top2 = []  # list of (score, code, templ_gray, candidate_gray, scale_factor)
    candidates = [gwarp, cv2.rotate(gwarp, cv2.ROTATE_180)]
    
    scales_to_try = MULTI_SCALE_FACTORS if USE_MULTI_SCALE else [1.0]
    
    # Optimize: Only use one template per card for initial matching (much faster)
    # Then use all templates for tie-breaking if needed
    if USE_MULTI_TEMPLATES and len(bank) > 52:
        # Group by card code and use first template of each
        from collections import OrderedDict
        unique_cards = OrderedDict()
        for code, templ in bank:
            if code not in unique_cards:
                unique_cards[code] = templ
        initial_bank = list(unique_cards.items())
    else:
        initial_bank = bank
    
    for g in candidates:
        for code, templ in initial_bank:
            # Try multiple scales for better off-center detection
            for scale in scales_to_try:
                if scale != 1.0:
                    # Scale the candidate image
                    new_h, new_w = int(H * scale), int(W * scale)
                    if new_h < 50 or new_w < 50 or new_h > H*2 or new_w > W*2:
                        continue  # Skip extreme scales
                    g_scaled = cv2.resize(g, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    # Resize back to template size for matching
                    g_scaled = cv2.resize(g_scaled, (W, H), interpolation=cv2.INTER_AREA)
                else:
                    g_scaled = g
                
                s = cv2.matchTemplate(g_scaled, templ, cv2.TM_CCOEFF_NORMED)[0][0]
                
                if len(top2) < 2:
                    top2.append((s, code, templ, g_scaled, scale))
                    top2.sort(reverse=True, key=lambda x: x[0])
                else:
                    if s > top2[-1][0]:
                        top2[-1] = (s, code, templ, g_scaled, scale)
                        top2.sort(reverse=True, key=lambda x: x[0])

    if not top2:
        return None, 0.0
    if len(top2) == 1:
        return top2[0][1], float(top2[0][0])

    # 2) tie-break for close matches, especially when distinguishing similar cards
    (s1, c1, t1, g1, _), (s2, c2, t2, g2, _) = top2
    R1, S1 = c1[:-1], c1[-1]
    R2, S2 = c2[:-1], c2[-1]

    close = (s1 - s2) <= 0.02             # margin you can tune
    same_rank = (R1 == R2)
    different_suit = (S1 != S2)
    same_suit = (S1 == S2)
    small_ranks = {R1, R2}.issubset({"2","3","4","5"})  # Include rank 5
    medium_ranks = {R1, R2}.issubset({"4","5","6","7"})  # Ranks that can be confused
    red_suit_confusion = {S1, S2} == {"H", "D"}  # Hearts vs Diamonds confusion
    clubs_diamonds_confusion = {S1, S2} == {"C", "D"}  # Clubs vs Diamonds confusion

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
    
    # Case 1b: Same rank, Clubs vs Diamonds confusion - use suit center ROI with tighter threshold
    elif close and same_rank and different_suit and clubs_diamonds_confusion:
        suit_rect = _suit_center_roi_rect(H, W)
        gs1 = _roi(g1, suit_rect); ts1 = _roi(t1, suit_rect)
        gs2 = _roi(g2, suit_rect); ts2 = _roi(t2, suit_rect)
        s1s = cv2.matchTemplate(gs1, ts1, cv2.TM_CCOEFF_NORMED)[0][0]
        s2s = cv2.matchTemplate(gs2, ts2, cv2.TM_CCOEFF_NORMED)[0][0]
        # Use slightly larger threshold for C vs D since they can be harder to distinguish
        if s2s > s1s + 0.005:  # 0.5% difference threshold
            return c2, float(s2)
        # else fall through to c1
    
    # Case 2: Same rank, different suit (general case) - use suit center ROI
    elif close and same_rank and different_suit:
        suit_rect = _suit_center_roi_rect(H, W)
        gs1 = _roi(g1, suit_rect); ts1 = _roi(t1, suit_rect)
        gs2 = _roi(g2, suit_rect); ts2 = _roi(t2, suit_rect)
        s1s = cv2.matchTemplate(gs1, ts1, cv2.TM_CCOEFF_NORMED)[0][0]
        s2s = cv2.matchTemplate(gs2, ts2, cv2.TM_CCOEFF_NORMED)[0][0]
        # Use a more robust threshold - require at least 0.3% difference to switch
        if s2s > s1s + 0.003:
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


def classify_fullcard(warp_bgr, score_thresh=None):
    """
    Returns (code, score). code like 'QS', '10D', or None if below threshold.
    Now supports multiple templates per card and improved preprocessing.
    """
    try:
        if score_thresh is None:
            score_thresh = float(os.getenv("FULLCARD_MIN_SCORE", "0.60"))
        bank = _load_templates()
        if not bank:
            return None, 0.0

        # Get target size for preprocessing
        target_size = (bank[0][1].shape[1], bank[0][1].shape[0])
        g = _prep(warp_bgr, target_size=target_size)
        
        # Fast path: only try 0° rotation first, if score is good enough, skip 180°
        code, score = _best_match_fast(g, bank)
        if score >= score_thresh:
            return code, score
        
        # If score is low, try 180° rotation
        g180 = cv2.rotate(g, cv2.ROTATE_180)
        code180, score180 = _best_match_fast(g180, bank)
        if score180 > score:
            return code180, score180
        
        if score < score_thresh:
            return None, max(score, score180)
        return code, score
    except Exception as e:
        # Only log errors if not in quiet mode
        if not int(os.getenv("QUIET_LOGS", "1")):
            print(f"[RANK_SUIT ERROR] classify_fullcard: {e}")
            import traceback
            traceback.print_exc()
        return None, 0.0


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

