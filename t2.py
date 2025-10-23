#!/usr/bin/env python3
# pip_counter_from_frame.py
# Detect cards, warp, and count center pips (no corners) + safe side-by-side viz

import cv2
import numpy as np
from typing import Tuple, List

# --- Tunables ---
WARP_SIZE: Tuple[int, int] = (400, 560)   # (w,h) portrait
CENTER_CROP_FRAC = 0.12                   # crop % from each side before counting
AREA_MIN, AREA_MAX = 700, 9000   # was 1200, 8000
INK_FRAC_FACE_GUARD = 0.22                # ink coverage threshold => face/unknown
SHOW_WARPS = True
IMAGE_PATH = "test_frames/frame2.jpg"


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Return TL, TR, BR, BL ordering for 4 points."""
    pts = pts.reshape(-1, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def warp_card(bgr: np.ndarray, cnt: np.ndarray, out_wh=WARP_SIZE) -> np.ndarray:
    """Perspective-warp a card contour to a portrait canvas."""
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect).astype(np.float32)
    ow, oh = out_wh
    w_side, h_side = rect[1]
    if w_side > h_side:  # enforce portrait
        ow, oh = oh, ow
    M = cv2.getPerspectiveTransform(_order_quad(box),
                                    np.array([[0, 0], [ow - 1, 0], [ow - 1, oh - 1], [0, oh - 1]], np.float32))
    return cv2.warpPerspective(bgr, M, (ow, oh))


def _central_crop(gray: np.ndarray, frac: float = CENTER_CROP_FRAC) -> np.ndarray:
    """Center crop a grayscale image by fractional margins."""
    h, w = gray.shape
    x0, x1 = int(w * frac), int(w * (1 - frac))
    y0, y1 = int(h * frac), int(h * (1 - frac))
    return gray[y0:y1, x0:x1]



def count_pips_center(
    gray,
    center_crop_frac: float = 0.12,     # include enough margin to keep outer pips
    area_min: int = 350,                # permissive lower bound (was higher)
    area_max: int = 15000,              # allow big pips in close warps
    close_k: int = 3,
    open_k: int = 2,
    dilate_k: int = 2,                  # gentle reconnect
    circ_min: float = 0.35,             # 4πA/P^2  (0..1); borders/art are usually <0.2
    circ_max: float = 0.88,             # avoid huge thick blobs
    min_center_gap_frac: float = 0.55   # de-dup: centers closer than ~0.55*avg(min(w,h)) merge
) -> int | None:
    """
    Count suit pips using only the center of a warped card (no corners).
    Returns 2..10 for number cards; None for face/unknown/ambiguous.
    Expects a grayscale warp ~400x560 (portrait).
    """
    import cv2, numpy as np, math

    H, W = gray.shape
    x0, x1 = int(W * center_crop_frac), int(W * (1 - center_crop_frac))
    y0, y1 = int(H * center_crop_frac), int(H * (1 - center_crop_frac))
    roi = gray[y0:y1, x0:x1]

    # Threshold ink
    roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)
    _, binv = cv2.threshold(roi_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphology
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    mask = cv2.morphologyEx(binv, cv2.MORPH_CLOSE, k_close, iterations=1)
    mask = cv2.morphologyEx(mask,  cv2.MORPH_OPEN,  k_open,  iterations=1)
    if dilate_k > 0:
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k)), iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- first pass: geometry & shape filters ---
    candidates = []
    roi_h, roi_w = mask.shape
    for c in cnts:
        a = cv2.contourArea(c)
        if a < area_min or a > area_max:
            continue

        x, y, w, h = cv2.boundingRect(c)

        # 1) Edge reject: touching ROI border? drop it (likely inner border/partial)
        if x <= 0 or y <= 0 or (x + w) >= roi_w - 1 or (y + h) >= roi_h - 1:
            continue

        # 2) Aspect & solidity (compact blobs)
        ar = (w / float(h)) if h else 0.0
        if not (0.5 <= ar <= 1.8):
            continue
        hull = cv2.convexHull(c)
        solidity = a / (cv2.contourArea(hull) + 1e-6)
        if solidity < 0.75:
            continue

        # 3) Circularity (filters thin borders/art fragments)
        p = cv2.arcLength(c, True)
        circ = (4.0 * math.pi * a) / (p * p + 1e-6)
        if not (circ_min <= circ <= circ_max):
            continue

        # Keep with centroid & a "scale" (min side of bbox) for de-dup step
        M = cv2.moments(c)
        cx = (M["m10"] / (M["m00"] + 1e-6))
        cy = (M["m01"] / (M["m00"] + 1e-6))
        candidates.append({"c": c, "area": a, "cx": cx, "cy": cy, "scale": min(w, h)})

    if not candidates:
        return None

    # --- de-dup: merge blobs with very close centers (split pips) ---
    candidates.sort(key=lambda d: d["area"], reverse=True)
    taken = []
    for d in candidates:
        too_close = False
        for k in taken:
            # scale-aware threshold
            gap_thresh = min(d["scale"], k["scale"]) * min_center_gap_frac
            if (abs(d["cx"] - k["cx"])**2 + abs(d["cy"] - k["cy"])**2) ** 0.5 < gap_thresh:
                too_close = True
                break
        if not too_close:
            taken.append(d)

    n = len(taken)

    # Final sanity: total pip ink coverage should be moderate
    total_ink = sum(d["area"] for d in taken) / float(roi_w * roi_h)
    if 2 <= n <= 10 and 0.015 <= total_ink <= 0.25:
        return n

    return None




def find_card_contours(bgr: np.ndarray) -> List[np.ndarray]:
    """Find likely card contours on a dark background."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    _, th = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN,  k, iterations=1)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cards = []
    H, W = gray.shape
    frame_area = H * W
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 0.002 * frame_area or area > 0.30 * frame_area:
            continue
        rect = cv2.minAreaRect(c)
        w, h = rect[1]
        if min(w, h) < 40:
            continue
        hull = cv2.convexHull(c)
        solidity = area / (cv2.contourArea(hull) + 1e-6)
        if solidity < 0.9:
            continue
        cards.append(c)
    return cards


def main():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(IMAGE_PATH)

    display = img.copy()
    contours = find_card_contours(img)

    results = []
    for i, c in enumerate(contours):
        warp = warp_card(img, c, out_wh=WARP_SIZE)
        gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
        n = count_pips_center(gray)
        results.append(n)

        # Draw result on original frame
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect).astype(int)
        cv2.drawContours(display, [box], -1, (0, 255, 0), 2)
        label = f"{n if n is not None else 'face/unk'}"
        cx, cy = np.mean(box[:, 0]), np.mean(box[:, 1])
        cv2.putText(display, label, (int(cx) - 20, int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        if SHOW_WARPS:
            # Build a same-size mask visualization and hstack safely
            roi = _central_crop(gray, frac=CENTER_CROP_FRAC)
            _, m = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close, iterations=1)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k_open,  iterations=1)
            m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2)), iterations=1)


            # Resize mask to warp's height/width before concatenation
            vis_h, vis_w = warp.shape[:2]
            m_resized = cv2.resize(m, (vis_w, vis_h), interpolation=cv2.INTER_NEAREST)
            m_bgr = cv2.cvtColor(m_resized, cv2.COLOR_GRAY2BGR)

            panel = np.hstack([warp, m_bgr])
            cv2.imshow(f"warp_{i} (pips={label})", panel)
            cv2.waitKey(0)
            cv2.destroyWindow(f"warp_{i} (pips={label})")

    print("Detected pip counts (per detected card):", results)

    cv2.imshow("frame with counts", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

