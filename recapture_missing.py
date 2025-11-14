#!/usr/bin/env python3
# Recapture missing sample #7 (45° rotation) for specific cards
import os, re, cv2, numpy as np
from pathlib import Path
import subprocess
import argparse
import time

# Load environment variables
try:
    from dotenv import load_dotenv
    if os.path.exists("env"):
        load_dotenv("env")
except Exception:
    pass

OUT_DIR  = Path("./templates_temp")
TEMPL_DIR = Path("./templates")
SIZE     = (int(os.getenv("WARP_W","500")), int(os.getenv("WARP_H","700")))
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))

# Import card detection functions (same as make_templates_cli.py)
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

def _warp_from_contour(img_bgr, cnt, out_size=SIZE):
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

ROI = tuple(int(p or "0") for p in os.getenv("ROI", "0,0,0,0").split(",")[:4]) if os.getenv("ROI") else (0,0,0,0)
CARD_AABB_MIN_FRAC = float(os.getenv("CARD_AABB_MIN_FRAC","0.0018"))
CARD_AABB_MAX_FRAC = float(os.getenv("CARD_AABB_MAX_FRAC","0.30"))
CARD_SHORTSIDE_MIN = int(os.getenv("CARD_SHORTSIDE_MIN","90"))
ASPECT_MIN, ASPECT_MAX = float(os.getenv("ASPECT_MIN","1.25")), float(os.getenv("ASPECT_MAX","1.75"))
SOLIDITY_MIN = float(os.getenv("SOLIDITY_MIN","0.88"))
FILL_OBB_MIN = float(os.getenv("FILL_OBB_MIN","0.80"))
ANGLE_TOL_DEG = float(os.getenv("ANGLE_TOL_DEG","16"))

def _card_candidates(mask, frame_w, frame_h):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 3000: continue
        x,y,w,h = cv2.boundingRect(cnt)
        aabb_frac = (w*h)/float(frame_w*frame_h + 1e-6)
        if not (CARD_AABB_MIN_FRAC <= aabb_frac <= CARD_AABB_MAX_FRAC): continue
        if min(w,h) < CARD_SHORTSIDE_MIN: continue
        hull = cv2.convexHull(cnt)
        solidity = area / (cv2.contourArea(hull)+1e-6)
        (cx,cy),(mw,mh),ang = cv2.minAreaRect(cnt)
        asp = max(mw,mh)/max(1.0,min(mw,mh))
        obb_fill = area / (mw*mh + 1e-6)
        if not (ASPECT_MIN <= asp <= ASPECT_MAX): continue
        if solidity < SOLIDITY_MIN or obb_fill < FILL_OBB_MIN: continue
        quad = cv2.boxPoints(cv2.minAreaRect(cnt))
        if not _right_angle_quad(quad, ANGLE_TOL_DEG): continue
        yield cnt

def normalize(img_bgr):
    """Normalize and resize image for template."""
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, SIZE, interpolation=cv2.INTER_AREA)
    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    return g

def play_chime():
    """Play capture chime."""
    try:
        subprocess.Popen(['afplay', '/System/Library/Sounds/Glass.aiff'], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

def mouse_callback(event, x, y, flags, param):
    """Mouse callback to capture image on click (debounced by 1 second)."""
    if event == cv2.EVENT_LBUTTONDOWN:
        now = time.time()
        last_click = param.get('last_click_time', 0)
        if now - last_click >= 1.0:  # 1 second debounce
            param['capture'] = True
            param['last_click_time'] = now

def main():
    # Cards missing sample #7 (45° rotation)
    missing_cards = ["10D", "2H", "3D", "3S", "5H", "6S", "QS"]
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Open camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[!] Could not open camera {CAMERA_INDEX}")
        return
    
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    except Exception:
        pass
    
    print("=" * 60)
    print("Recapture Missing Sample #7 (45° rotation)")
    print("=" * 60)
    print(f"Recapturing sample #7 for {len(missing_cards)} cards:")
    for card in missing_cards:
        print(f"  - {card}")
    print("=" * 60)
    print("Controls:")
    print("  - Mouse click: Capture")
    print("  - Q key: Quit/Skip")
    print("=" * 60)
    
    window_name = "Recapture Missing - Click to capture, Q to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    capture_state = {'capture': False, 'last_click_time': 0}
    cv2.setMouseCallback(window_name, mouse_callback, capture_state)
    
    card_idx = 0
    last_warp = None
    
    while card_idx < len(missing_cards):
        current_card = missing_cards[card_idx]
        out_file = OUT_DIR / f"{current_card}7.png"
        
        # Load existing template for reference
        template_img = None
        template_path = TEMPL_DIR / f"{current_card}.png"
        if template_path.exists():
            template_img = cv2.imread(str(template_path))
        
        print(f"\n[{card_idx + 1}/{len(missing_cards)}] Card: {current_card} - Sample #7 (45° rotation)")
        print(f"  Will save as: {out_file.name}")
        print(f"  Rotate card ~45° in center, click to capture")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[!] Failed to read from camera")
                break
            
            # Preprocess frame
            proc, offx, offy = _roi(frame, ROI) if sum(ROI) != 0 else (frame, 0, 0)
            gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
            norm = _normalize_L(gray)
            _, th = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Detect cards
            detected_warp = None
            display_frame = frame.copy()
            
            for cnt in _card_candidates(th, proc.shape[1], proc.shape[0]):
                warp, box, (mw, mh) = _warp_from_contour(proc, cnt)
                detected_warp = warp
                last_warp = warp
                
                # Draw detection box
                box_int = box.astype(np.int32)
                cv2.drawContours(display_frame, [box_int + [offx, offy]], -1, (0, 255, 0), 2)
                break
            
            # Draw overlay
            h, w = display_frame.shape[:2]
            base_font_scale = max(w / 400.0, h / 300.0)
            font = base_font_scale * 1.5
            thickness = max(4, int(base_font_scale * 2))
            
            # Position number
            pos_text = "7"
            text_x = int(w * 0.1)
            text_y = int(h * 0.25)
            cv2.putText(display_frame, pos_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font, (0, 0, 0), thickness + 4)
            cv2.putText(display_frame, pos_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font, (0, 255, 0), thickness)
            
            # Card name
            card_text = f"Card: {current_card} - Sample 7 (45° rotation)"
            cv2.putText(display_frame, card_text, (text_x, int(h * 0.35)), 
                       cv2.FONT_HERSHEY_SIMPLEX, font * 0.7, (255, 255, 255), thickness)
            
            # Status
            status = "Card detected - Click to capture!" if detected_warp is not None else "No card detected"
            status_color = (0, 255, 0) if detected_warp is not None else (255, 200, 200)
            cv2.putText(display_frame, status, (text_x, int(h * 0.4)), 
                       cv2.FONT_HERSHEY_SIMPLEX, font * 0.6, status_color, thickness)
            
            # Show template in top right
            if template_img is not None:
                template_display_h = int(h * 0.25)
                template_aspect = template_img.shape[1] / template_img.shape[0]
                template_display_w = int(template_display_h * template_aspect)
                template_display = cv2.resize(template_img, (template_display_w, template_display_h))
                corner_x = w - template_display_w - 20
                corner_y = 20
                template_bg = display_frame[corner_y:corner_y+template_display_h, corner_x:corner_x+template_display_w].copy()
                cv2.rectangle(template_bg, (0, 0), (template_display_w, template_display_h), (0, 0, 0), -1)
                display_frame[corner_y:corner_y+template_display_h, corner_x:corner_x+template_display_w] = cv2.addWeighted(
                    template_bg, 0.3, template_display, 0.7, 0)
                cv2.rectangle(display_frame, (corner_x-2, corner_y-2), 
                             (corner_x+template_display_w+2, corner_y+template_display_h+2), (255, 255, 0), 3)
            
            # Show detected warp in bottom right
            if detected_warp is not None:
                warp_display = cv2.resize(detected_warp, (200, 280))
                h_w, w_w = warp_display.shape[:2]
                warp_x = w - w_w - 20
                warp_y = h - h_w - 20
                display_frame[warp_y:warp_y+h_w, warp_x:warp_x+w_w] = warp_display
                cv2.rectangle(display_frame, (warp_x-2, warp_y-2), (warp_x+w_w+2, warp_y+h_w+2), (0, 255, 0), 3)
            
            cv2.imshow(window_name, display_frame)
            
            # Check for mouse capture
            if capture_state['capture']:
                capture_state['capture'] = False
                
                warp_to_save = last_warp if last_warp is not None else frame
                
                if warp_to_save is not None:
                    normalized = normalize(warp_to_save)
                    cv2.imwrite(str(out_file), normalized)
                    print(f"  ✓ Saved: {out_file.name}")
                    play_chime()
                    card_idx += 1
                    break
                else:
                    print(f"  ⚠ No card detected - nothing saved. Try again.")
            
            # Check for keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("\nQuitting...")
                break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone! Recaptured {card_idx} out of {len(missing_cards)} missing samples")

if __name__ == "__main__":
    main()

