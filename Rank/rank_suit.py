# rank_suit.py — classic CV rank/suit reader (no training, local-only) + debug CLI + auto-rotate
import os, sys, cv2, glob, numpy as np

# ============ config ============
TEMPL_DIR = os.getenv("CARD_TEMPL_DIR", "./card_templates")   # rank_*.png, suit_*.png
INDEX_UL = (0.02, 0.02, 0.20, 0.26)  # (x,y,w,h) fractions for upper-left index
INDEX_BR = (0.78, 0.72, 0.20, 0.26)  # symmetric for 180° rotation
SIZE = (48, 48)
RANK_MIN_SCORE = float(os.getenv("RANK_MIN_SCORE", "0.45"))
SUIT_MIN_SCORE = float(os.getenv("SUIT_MIN_SCORE", "0.45"))
MARGIN_MIN     = float(os.getenv("NCC_MARGIN_MIN", "0.05"))   # top vs second-best

_rank_tmpls, _suit_tmpls = {}, {}

# ---------- utils ----------
def _ensure_templates():
    global _rank_tmpls, _suit_tmpls
    if _rank_tmpls and _suit_tmpls: return

    def _prep_gray_binarize(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim==3 else img
        g = cv2.resize(g, SIZE, interpolation=cv2.INTER_AREA)
        g = cv2.GaussianBlur(g, (3,3), 0)
        _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # prefer dark glyph on bright background
        if bw.mean() < 127: bw = 255 - bw
        return bw

    for p in glob.glob(os.path.join(TEMPL_DIR, "rank_*.png")):
        name = os.path.splitext(os.path.basename(p))[0].split("_", 1)[1]
        _rank_tmpls[name] = _prep_gray_binarize(cv2.imread(p, cv2.IMREAD_GRAYSCALE))
    for p in glob.glob(os.path.join(TEMPL_DIR, "suit_*.png")):
        name = os.path.splitext(os.path.basename(p))[0].split("_", 1)[1]
        _suit_tmpls[name] = _prep_gray_binarize(cv2.imread(p, cv2.IMREAD_GRAYSCALE))

def _ncc(a, b):
    a = a.astype(np.float32); b = b.astype(np.float32)
    am, bm = a.mean(), b.mean()
    num = ((a-am)*(b-bm)).sum()
    den = np.sqrt(((a-am)**2).sum() * ((b-bm)**2).sum()) + 1e-6
    return float(num/den)

def _prep_bw(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim==3 else img
    g = cv2.resize(g, SIZE, interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(g, (3,3), 0)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if bw.mean() < 127: bw = 255 - bw
    return bw

def _roi_frac(img, frac):
    h, w = img.shape[:2]
    fx, fy, fw, fh = frac
    x = int(fx*w); y = int(fy*h); ww = int(fw*w); hh = int(fh*w if False else fh*h)  # keep aspect as fraction of h
    hh = int(fh*h)
    return img[y:y+hh, x:x+ww].copy()

def _split_rank_suit(roi_bgr):
    # split index block (rank above, suit below) via horizontal valley
    g = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3,3), 0)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if bw.mean() < 127: bw = 255 - bw
    h, w = bw.shape
    ink = (255 - bw).sum(axis=1)
    # search mid band for a valley between two text blobs
    lo, hi = int(0.35*h), int(0.75*h)
    cut = min(range(lo, hi), key=lambda y: ink[y]) if hi > lo else h//2
    return roi_bgr[:cut, :], roi_bgr[cut:, :]

def suit_symbol(name):
    return {"spade":"♠","heart":"♥","diamond":"♦","club":"♣"}.get(name, "?")

# ---------- core matcher (now tries both orientations) ----------
def _classify_from_orientation(warp_bgr, frac):
    roi = _roi_frac(warp_bgr, frac)
    if roi.size == 0: 
        return None
    rroi, sroi = _split_rank_suit(roi)
    rbw, sbw = _prep_bw(rroi), _prep_bw(sroi)

    r_scores = sorted(((name, _ncc(rbw, tmpl)) for name, tmpl in _rank_tmpls.items()),
                      key=lambda x: x[1], reverse=True)
    s_scores = sorted(((name, _ncc(sbw, tmpl)) for name, tmpl in _suit_tmpls.items()),
                      key=lambda x: x[1], reverse=True)

    rbest, rscore = (r_scores[0] if r_scores else (None, -1.0))
    r2 = r_scores[1][1] if len(r_scores) > 1 else -1.0
    sbest, sscore = (s_scores[0] if s_scores else (None, -1.0))
    s2 = s_scores[1][1] if len(scores := s_scores) > 1 else -1.0

    ok = (rscore >= RANK_MIN_SCORE and sscore >= SUIT_MIN_SCORE and
          (rscore - r2) >= MARGIN_MIN and (sscore - s2) >= MARGIN_MIN)
    return {
        "ok": ok, "rank": rbest, "suit": sbest,
        "rscore": rscore, "sscore": sscore, "r2": r2, "s2": s2,
        "roi": roi, "rbw": rbw, "sbw": sbw
    }

def classify_rank_suit_from_warp(warp_bgr):
    """
    Input: a single warped card image (your warp from perspective transform)
    Returns: (rank, suit, rscore, sscore) or (None, None, _, _) if low confidence
    NOTE: This version auto-tries 0° and 180° and uses both UL and BR index crops.
    """
    _ensure_templates()
    if not _rank_tmpls or not _suit_tmpls:
        return None, None, 0.0, 0.0

    # try both orientations
    candidates = []
    for orient_name, img in (("orig", warp_bgr),
                             ("rot180", cv2.rotate(warp_bgr, cv2.ROTATE_180))):
        for frac in (INDEX_UL, INDEX_BR):
            res = _classify_from_orientation(img, frac)
            if res is None: 
                continue
            res["orient"] = orient_name
            res["frac"] = frac
            # store even if not "ok" so we can pick the best overall sum
            candidates.append(res)

    if not candidates:
        return None, None, 0.0, 0.0

    # pick best by r+ s scores, but enforce thresholds
    best = max(candidates, key=lambda d: d["rscore"] + d["sscore"])
    if best["ok"]:
        return best["rank"], best["suit"], best["rscore"], best["sscore"]
    return None, None, best["rscore"], best["sscore"]

# ---------- Debug CLI ----------
def _draw_preview(warp_bgr, best):
    # Compose a visual showing: warp (possibly rotated), ROI, rbw/sbw and scores
    img = cv2.rotate(warp_bgr, cv2.ROTATE_180) if best.get("orient") == "rot180" else warp_bgr.copy()
    roi = best["roi"]; rbw = best["rbw"]; sbw = best["sbw"]
    # layout
    def to3(x): return cv2.cvtColor(x, cv2.COLOR_GRAY2BGR) if len(x.shape)==2 else x
    def resize_h(x, h): 
        hh, ww = x.shape[:2]; return cv2.resize(x, (int(ww*(h/float(hh))), h))
    top = np.hstack([resize_h(img, 280), resize_h(to3(roi), 280)])
    tilesz = 220
    bot = np.hstack([cv2.resize(to3(rbw), (tilesz, tilesz)),
                     cv2.resize(to3(sbw), (tilesz, tilesz))])
    pad = 12
    W = max(top.shape[1], bot.shape[1])
    canvas = np.full((top.shape[0]+pad+bot.shape[0], W, 3), 255, np.uint8)
    canvas[0:top.shape[0], 0:top.shape[1]] = top
    canvas[top.shape[0]+pad: top.shape[0]+pad+bot.shape[0], 0:bot.shape[1]] = bot

    # Write text
    rname = best['rank'] or "?"
    sname = best['suit'] or "?"
    txt1 = f"{rname} / {sname}  r={best['rscore']:.3f} (Δ{best['rscore']-best['r2']:.3f})  s={best['sscore']:.3f} (Δ{best['sscore']-best['s2']:.3f})"
    cv2.putText(canvas, txt1, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    cv2.putText(canvas, txt1, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40,40,255), 1)
    return canvas

def _debug_one(path):
    _ensure_templates()
    img = cv2.imread(path)
    if img is None:
        print(f"[!] unreadable: {path}")
        return

    # evaluate both orientations & both corners with internals to get details
    cand = []
    for orient_name, img_o in (("orig", img), ("rot180", cv2.rotate(img, cv2.ROTATE_180))):
        for frac in (INDEX_UL, INDEX_BR):
            res = _classify_from_orientation(img_o, frac)
            if res is None: continue
            res["orient"] = orient_name; res["frac"] = frac
            cand.append(res)
    if not cand:
        print("[!] no ROI found"); return

    best = max(cand, key=lambda d: d["rscore"] + d["sscore"])
    r, s = best["rank"], best["suit"]

    ok = best["ok"]
    pretty = f"{r}{suit_symbol(s)}" if (r and s) else "?"
    print(f"Best: {pretty}  r={best['rscore']:.3f} (Δ{best['rscore']-best['r2']:.3f})  "
          f"s={best['sscore']:.3f} (Δ{best['sscore']-best['s2']:.3f})  "
          f"orient={best['orient']} corner={'UL' if best['frac']==INDEX_UL else 'BR'}  ok={ok}")

    prev = _draw_preview(img, best)
    outp = os.path.splitext(path)[0] + "_preview.png"
    cv2.imwrite(outp, prev)
    try:
        cv2.imshow("rank_suit debug", prev)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        pass
    print(f"[saved] {outp}")

# ---------- CLI entry ----------
if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1].lower() == "debug":
        _debug_one(sys.argv[2])
    else:
        # no-op CLI to avoid accidental prints when imported
        pass

