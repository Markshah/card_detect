#!/usr/bin/env python3
# make_full_templates_cli.py — label warps into ./templates as <RANK><SUIT>.png
import os, re, cv2
from pathlib import Path

WARP_DIR = Path("./debug/warps_raw")
OUT_DIR  = Path("./templates")
SIZE     = (int(os.getenv("WARP_W","400")), int(os.getenv("WARP_H","560")))

OUT_DIR.mkdir(parents=True, exist_ok=True)

RANK_MAP = {
    "A":"A","ACE":"A","1":"A",
    "K":"K","KING":"K","Q":"Q","QUEEN":"Q","J":"J","JACK":"J",
    "10":"10","T":"10","TEN":"10","9":"9","8":"8","7":"7","6":"6",
    "5":"5","4":"4","3":"3","2":"2"
}
SUIT_MAP = {
    "S":"S","SPADE":"S","SPADES":"S",
    "H":"H","HEART":"H","HEARTS":"H",
    "D":"D","DIAMOND":"D","DIAMONDS":"D",
    "C":"C","CLUB":"C","CLUBS":"C"
}

def parse_label(s: str):
    s = s.strip().upper()
    if not s: return None
    m = re.fullmatch(r"(A|K|Q|J|10|[2-9])\s*([SHDC])", s)
    if m: return m.group(1), m.group(2)
    toks = re.findall(r"[A-Z0-9]+", s)
    r = next((RANK_MAP.get(t) for t in toks if t in RANK_MAP), None)
    u = next((SUIT_MAP.get(t) for t in toks if t in SUIT_MAP), None)
    return (r,u) if (r and u) else None

def normalize(img_bgr):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, SIZE, interpolation=cv2.INTER_AREA)
    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    return g

def main():
    warps = sorted(list(WARP_DIR.glob("*.png")) + list(WARP_DIR.glob("*.jpg")))
    if not warps:
        print(f"[!] No warps in {WARP_DIR}. Run card_detect.py with SAVE_WARP_RAW=1.")
        return
    print("Template builder (full-card). Enter like: as, 10h, qd, 'ace of spades'. Enter=skip, q=quit.")
    cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
    for i, p in enumerate(warps, 1):
        img = cv2.imread(str(p))
        if img is None: continue
        disp = cv2.resize(img, (int(img.shape[1]*0.6), int(img.shape[0]*0.6)))
        cv2.imshow("preview", disp)
        print(f"[{i}/{len(warps)}] {p.name}")
        s = input("  Card: ").strip()
        if s.lower() in ("q","quit","exit"): break
        if not s: continue
        parsed = parse_label(s)
        if not parsed:
            print("  (ignored) couldn't parse. Examples: as, 10h, queen of diamonds")
            continue
        r, su = parsed
        out = OUT_DIR / f"{r}{su}.png"
        if out.exists():
            print(f"  {out.name} already present (skipped)")
            continue
        cv2.imwrite(str(out), normalize(img))
        print(f"  saved {out.name}")
        cv2.waitKey(1)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

