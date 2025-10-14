#!/usr/bin/env python3
import os, cv2, glob, numpy as np, subprocess, re

TEMPL_DIR = os.getenv("CARD_TEMPL_DIR", "./card_templates")
WARP_DIR  = os.getenv("WARP_RAW_DIR", "./debug/warps_raw")
PREVIEW   = int(os.getenv("PREVIEW", "1"))  # 1 = auto-open preview image (macOS 'open')

INDEX_UL  = (0.02, 0.02, 0.20, 0.26)
INDEX_BR  = (0.78, 0.72, 0.20, 0.26)
SIZE = (48, 48)

NEED_RANKS = {"A","2","3","4","5","6","7","8","9","10","J","Q","K"}
NEED_SUITS = {"spade","heart","diamond","club"}


RANK_SYNONYMS = {
    "A":"A","ACE":"A",
    "K":"K","KING":"K",
    "Q":"Q","QUEEN":"Q",
    "J":"J","JACK":"J",
    "T":"10","10":"10","TEN":"10",
    "2":"2","TWO":"2",
    "3":"3","THREE":"3",
    "4":"4","FOUR":"4",
    "5":"5","FIVE":"5",
    "6":"6","SIX":"6",
    "7":"7","SEVEN":"7",
    "8":"8","EIGHT":"8",
    "9":"9","NINE":"9",
}

SUIT_SYNONYMS = {
    "S":"spade","SPADE":"spade","SPADES":"spade","♠":"spade",
    "H":"heart","HEART":"heart","HEARTS":"heart","♥":"heart",
    "D":"diamond","DIAMOND":"diamond","DIAMONDS":"diamond","♦":"diamond",
    "C":"club","CLUB":"club","CLUBS":"club","♣":"club",
}

def roi_frac(img, frac):
    h, w = img.shape[:2]; fx, fy, fw, fh = frac
    x = int(fx*w); y = int(fy*h); ww = int(fw*w); hh = int(fh*h)
    return img[y:y+hh, x:x+ww].copy()

def prep_bw(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim==3 else img
    g = cv2.resize(g, SIZE, interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(g, (3,3), 0)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if bw.mean() < 127: bw = 255 - bw
    return bw

def split_rank_suit(roi_bgr):
    g = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3,3), 0)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if bw.mean() < 127: bw = 255 - bw
    h = bw.shape[0]; ink = (255 - bw).sum(axis=1)
    lo, hi = int(0.35*h), int(0.75*h)
    if hi <= lo:  # tiny ROI safety
        cut = h//2
    else:
        cut = min(range(lo, hi), key=lambda y: ink[y])
    return roi_bgr[:cut, :], roi_bgr[cut:, :]

def best_corner(img):
    cand = []
    for frac in (INDEX_UL, INDEX_BR):
        roi = roi_frac(img, frac)
        if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
            continue
        rroi, sroi = split_rank_suit(roi)
        rbw, sbw = prep_bw(rroi), prep_bw(sroi)
        ink = (255 - rbw).sum() + (255 - sbw).sum()
        cand.append((ink, rbw, sbw, roi))
    if not cand:
        return None
    return max(cand, key=lambda t: t[0])  # (ink, rbw, sbw, roi_color)

# ---- preview helpers (fixed-height stacking) ----
def _resize_to_height(img, target_h):
    h, w = img.shape[:2]
    if h == target_h: return img
    new_w = int(round(w * (target_h / float(h))))
    return cv2.resize(img, (new_w, target_h))

def _to_bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img

def make_preview(img, rbw, sbw, roi_color):
    rbw_c = _to_bgr(rbw); sbw_c = _to_bgr(sbw); roi_bgr = _to_bgr(roi_color)
    warp_small = _resize_to_height(img, 280)
    roi_small  = _resize_to_height(roi_bgr, warp_small.shape[0])
    top = np.hstack([warp_small, roi_small])
    tile = 220
    rbw_c = cv2.resize(rbw_c, (tile, tile)); sbw_c = cv2.resize(sbw_c, (tile, tile))
    bot = np.hstack([rbw_c, sbw_c])
    pad = 12
    width = max(top.shape[1], bot.shape[1])
    canvas = np.full((top.shape[0]+pad+bot.shape[0], width, 3), 255, np.uint8)
    canvas[0:top.shape[0], 0:top.shape[1]] = top
    canvas[top.shape[0]+pad: top.shape[0]+pad+bot.shape[0], 0:bot.shape[1]] = bot
    return canvas

# ---- input parsing: "as", "A S", "ace of spades", "qd", "queen diamonds", "10h", etc. ----
def parse_combo(s: str):
    """
    Returns (rank, suit) or (None, None)
    Accepts: 'as', 'a s', 'ace spades', 'ace of spades', 'qd', 'q♦', '10h', 'ten hearts'
    Quit only on lowercase 'q' or 'quit'.
    """
    s = s.strip()
    if s.lower() in ("q", "quit"):
        return ("__QUIT__", None)

    # remove 'of' and punctuation, split by space or keep compact pairs
    cleaned = re.sub(r"[^\w\s♠♥♦♣]", " ", s)
    cleaned = re.sub(r"\bof\b", " ", cleaned, flags=re.I).strip()
    parts = cleaned.split()
    # If compact like 'as' or '10h', split into rank+suitchar
    if len(parts) == 1:
        tok = parts[0]
        # try suit symbol at end
        m = re.match(r"(?i)^(10|[AKQJT2-9])\s*([shdc♠♥♦♣])$", tok)
        if m:
            r_raw, su = m.group(1).upper(), m.group(2)
            rank = RANK_SYNONYMS.get(r_raw, None)
            suit = SUIT_SYNONYMS.get(su.upper(), SUIT_SYNONYMS.get(su, None))
            return (rank, suit)
        # try spelled out rank/suit in one token? unlikely—fall through
        # try two-char rank like 'as', 'qd', 'tc'
        if len(tok) in (2,3):  # allow '10h'
            if tok[:2].lower() == "10":
                r_raw = "10"; su = tok[2:] if len(tok) == 3 else ""
            else:
                r_raw = tok[0].upper()
                su = tok[1:]
            rank = RANK_SYNONYMS.get(r_raw, None)
            suit = SUIT_SYNONYMS.get(su.upper(), SUIT_SYNONYMS.get(su, None))
            return (rank, suit if suit in NEED_SUITS else None)
        # else treat as single word (e.g., 'queenhearts' -> split alpha groups)
        chunks = re.findall(r"(10|[AKQJT2-9]|ACE|KING|QUEEN|JACK|TEN|[A-Z]+|♠|♥|♦|♣)", tok, flags=re.I)
        if len(chunks) >= 2:
            return parse_combo(" ".join(chunks[:2]))

    # multi-token like "ace spades", "queen of diamonds", "10 hearts"
    if len(parts) >= 2:
        r_tok = parts[0].upper()
        s_tok = parts[1].upper()
        rank = RANK_SYNONYMS.get(r_tok, None)
        suit = SUIT_SYNONYMS.get(s_tok, SUIT_SYNONYMS.get(parts[1], None))
        if not rank and len(parts) >= 3:
            # maybe 'ten' 'of' 'hearts' got split already
            r_tok = parts[0].upper()
            s_tok = parts[-1].upper()
            rank = RANK_SYNONYMS.get(r_tok, None)
            suit = SUIT_SYNONYMS.get(s_tok, SUIT_SYNONYMS.get(parts[-1], None))
        return (rank, suit)

    return (None, None)

def main():
    os.makedirs(TEMPL_DIR, exist_ok=True)
    prev_dir = os.path.join(TEMPL_DIR, "_preview"); os.makedirs(prev_dir, exist_ok=True)

    warps = sorted(glob.glob(os.path.join(WARP_DIR, "*.png")) +
                   glob.glob(os.path.join(WARP_DIR, "*.jpg")))
    if not warps:
        print(f"[!] No warps found in {WARP_DIR}."); return

    have_rank = {os.path.basename(p).split("rank_")[-1].split(".")[0]
                 for p in glob.glob(os.path.join(TEMPL_DIR,"rank_*.png"))}
    have_suit = {os.path.basename(p).split("suit_")[-1].split(".")[0]
                 for p in glob.glob(os.path.join(TEMPL_DIR,"suit_*.png"))}

    print("Template builder with combo input. Examples: 'as', 'qd', '10h', 'ace of spades'.")
    print("Enter=skip, 'q' or 'quit' to stop.\n")

    for i, p in enumerate(warps, 1):
        img = cv2.imread(p)
        if img is None: 
            print(f" [skip] unreadable {p}")
            continue

        res = best_corner(img)
        if res is None:
            print(f" [skip] {os.path.basename(p)} (empty ROI)")
            continue
        ink, rbw, sbw, roi = res

        # Make & open preview
        prev = make_preview(img, rbw, sbw, roi)
        outp = os.path.join(prev_dir, f"prev_{i:03d}_{os.path.basename(p)}")
        cv2.imwrite(outp, prev)
        if PREVIEW:
            try: subprocess.Popen(["open", outp])  # macOS Preview
            except Exception: pass

        print(f"[{i}/{len(warps)}] {os.path.basename(p)}")
        ans = input("  Card (e.g., 'as', 'qd', '10h', 'ace of spades') [Enter=skip, q=quit]: ").strip()
        rank, suit = parse_combo(ans)

        if rank == "__QUIT__":
            break
        if not ans:
            continue
        if rank not in NEED_RANKS or suit not in NEED_SUITS:
            print("  (ignored) couldn't parse rank/suit. Try again next image.")
            continue

        # Save only what's missing so you can label quickly
        if rank not in have_rank:
            cv2.imwrite(os.path.join(TEMPL_DIR, f"rank_{rank}.png"), rbw)
            have_rank.add(rank)
            print(f"    saved rank_{rank}.png")
        else:
            print(f"    rank_{rank}.png already present")

        if suit not in have_suit:
            cv2.imwrite(os.path.join(TEMPL_DIR, f"suit_{suit}.png"), sbw)
            have_suit.add(suit)
            print(f"    saved suit_{suit}.png")
        else:
            print(f"    suit_{suit}.png already present")

        if len(have_rank) >= 13 and len(have_suit) >= 4:
            print("\n[*] All templates captured. Done.")
            break

    print(f"\nHave ranks: {sorted(have_rank)}")
    print(f"Have suits: {sorted(have_suit)}")

if __name__ == "__main__":
    main()

