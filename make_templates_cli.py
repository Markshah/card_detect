#!/usr/bin/env python3
# make_full_templates_cli.py — unified card detection + interactive template capture
import os, re, cv2, numpy as np
from pathlib import Path
import subprocess
import argparse

# Load environment variables - prioritize template-specific env, but fall back to main env for WARP_W/H
# This ensures WARP_W/H stay in sync with card_detect.py while allowing other settings to differ
try:
    from dotenv import load_dotenv
    # Load main env first (for WARP_W/H consistency)
    if os.path.exists("env"):
        load_dotenv("env")
    # Load template-specific env if it exists (overrides for template-specific settings)
    if os.path.exists("env_templates"):
        load_dotenv("env_templates", override=False)  # Don't override WARP_W/H from main env
except Exception:
    pass

OUT_DIR  = Path("./templates_temp")
TEMPL_DIR = Path("./templates")  # Existing templates folder
# WARP_W/H must match card_detect.py - read from main env file
SIZE     = (int(os.getenv("WARP_W","400")), int(os.getenv("WARP_H","560")))
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))

# Import card detection functions from card_detect.py
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# We'll need these from card_detect.py - importing key functions
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

# Card detection parameters (from card_detect.py env vars)
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

# Generate all 52 cards in order: A-K for each suit (S, H, D, C)
RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
SUITS = ["S", "H", "D", "C"]

def generate_all_cards():
    """Generate list of all 52 cards: AS, AH, AD, AC, 2S, 2H, 2D, 2C, ..., KS, KH, KD, KC"""
    cards = []
    for rank in RANKS:
        for suit in SUITS:
            cards.append(f"{rank}{suit}")
    return cards

def get_next_number(base_name):
    """Find next available number for a card (AS1, AS2, etc.)"""
    existing_numbers = []
    
    # Check base file (AS.png) - treat as version 0
    base_file = OUT_DIR / f"{base_name}.png"
    if base_file.exists():
        existing_numbers.append(0)
    
    # Check numbered versions (AS1.png, AS2.png, etc.)
    for existing_file in OUT_DIR.glob(f"{base_name}*.png"):
        name_no_ext = existing_file.stem  # e.g., "AS1"
        num_match = re.search(r"(\d+)$", name_no_ext)
        if num_match:
            existing_numbers.append(int(num_match.group(1)))
    
    if existing_numbers:
        return max(existing_numbers) + 1
    else:
        return 1

def normalize(img_bgr):
    """Normalize and resize image for template."""
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, SIZE, interpolation=cv2.INTER_AREA)
    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    return g

def play_chime(capture=False):
    """Play a chime sound on macOS. capture=True for capture chime, False for rank change chime."""
    try:
        if capture:
            # Short beep for capture
            subprocess.Popen(['afplay', '/System/Library/Sounds/Glass.aiff'], 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            # Different sound for rank change
            subprocess.Popen(['afplay', '/System/Library/Sounds/Ping.aiff'], 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass  # Silently fail if sound can't play

def mouse_callback(event, x, y, flags, param):
    """Mouse callback to capture image on click (debounced by 1 second)."""
    if event == cv2.EVENT_LBUTTONDOWN:
        import time
        now = time.time()
        last_click = param.get('last_click_time', 0)
        if now - last_click >= 1.0:  # 1 second debounce
            param['capture'] = True
            param['last_click_time'] = now

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Interactive template capture for playing cards')
    parser.add_argument('--start-rank', type=str, default='A', 
                       help='Rank to start from (A, 2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K). Default: A')
    args = parser.parse_args()
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate all 52 cards
    all_cards = generate_all_cards()
    
    # Skip to specified rank if provided
    start_rank = args.start_rank.upper()
    if start_rank not in RANKS:
        print(f"Warning: Invalid rank '{start_rank}'. Valid ranks: {', '.join(RANKS)}")
        print("Starting from beginning (A)")
    else:
        # Find first card of this rank (e.g., "3S" for rank "3")
        start_card = f"{start_rank}S"
        try:
            start_idx = all_cards.index(start_card)
            all_cards = all_cards[start_idx:]
            print(f"Starting from rank {start_rank} ({start_card}) - card {start_idx + 1}/52")
        except ValueError:
            print(f"Warning: Could not find {start_card}, starting from beginning")
    
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
    
    # Define 7 samples per card
    SAMPLE_NAMES = [
        "Center, normal orientation",
        "Left position",
        "Right position", 
        "Up position",
        "Down position",
        "Center, rotated ~15°",
        "Center, rotated ~45°"
    ]
    SAMPLES_PER_CARD = len(SAMPLE_NAMES)
    
    print("=" * 60)
    print("Unified Card Detection + Template Capture")
    print("=" * 60)
    print("Controls:")
    print("  - Mouse click: Capture current sample")
    print("  - Q key: Quit (or skip remaining samples)")
    print("")
    print(f"Capturing {SAMPLES_PER_CARD} samples per card:")
    for i, name in enumerate(SAMPLE_NAMES, 1):
        print(f"  {i}. {name}")
    print("=" * 60)
    
    window_name = "Template Capture - Click to capture, N for next, Q to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Shared state for mouse callback (with debounce)
    import time
    capture_state = {'capture': False, 'last_click_time': 0}
    cv2.setMouseCallback(window_name, mouse_callback, capture_state)
    
    card_idx = 0
    last_warp = None
    last_rank = None  # Track rank changes
    
    while card_idx < len(all_cards):
        current_card = all_cards[card_idx]
        current_rank = current_card[:-1]  # Extract rank (e.g., "A", "2", "3")
        sample_idx = 0
        
        # Play rank change chime if this is a new rank
        if last_rank is not None and current_rank != last_rank:
            play_chime(capture=False)  # Rank change chime
        last_rank = current_rank
        
        # Load existing template image for reference
        template_img = None
        template_path = TEMPL_DIR / f"{current_card}.png"
        if template_path.exists():
            template_img = cv2.imread(str(template_path))
            if template_img is not None:
                print(f"\n[{card_idx + 1}/52] Card: {current_card} (template found)")
            else:
                print(f"\n[{card_idx + 1}/52] Card: {current_card} (template not readable)")
        else:
            print(f"\n[{card_idx + 1}/52] Card: {current_card} (no template found)")
        
        print(f"  Starting {SAMPLES_PER_CARD} samples...")
        
        while sample_idx < SAMPLES_PER_CARD:
            sample_name = SAMPLE_NAMES[sample_idx]
            next_num = get_next_number(current_card)
            
            print(f"  Sample {sample_idx + 1}/{SAMPLES_PER_CARD}: {sample_name}")
            print(f"    Will save as: {current_card}{next_num}.png")
            print(f"    Position card and click mouse to capture")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[!] Failed to read from camera")
                    break
                
                # Preprocess frame (same as card_detect.py)
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
                    
                    # Draw detection box on frame
                    box_int = box.astype(np.int32)
                    cv2.drawContours(display_frame, [box_int + [offx, offy]], -1, (0, 255, 0), 2)
                    break  # Use first detected card
                
                # Draw overlay - just show position number
                h, w = display_frame.shape[:2]
                
                # Calculate font scale - moderate size for visibility
                base_font_scale = max(w / 400.0, h / 300.0)  # Scale based on both width and height
                large_font = base_font_scale * 1.5  # Position number - moderate size
                thickness_large = max(4, int(base_font_scale * 2))
                
                # Position number - very large in center/top area
                pos_text = f"{sample_idx + 1}"
                text_x = int(w * 0.1)  # 10% from left
                text_y = int(h * 0.25)  # 25% from top
                
                # Draw black outline first for contrast
                cv2.putText(display_frame, pos_text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, large_font, (0, 0, 0), thickness_large + 4)
                # Then draw main text in bright green
                cv2.putText(display_frame, pos_text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, large_font, (0, 255, 0), thickness_large)
                
                # Show existing template image in top right corner
                if template_img is not None:
                    # Resize template to fit in corner
                    template_display_h = int(h * 0.25)  # 25% of screen height
                    template_aspect = template_img.shape[1] / template_img.shape[0]
                    template_display_w = int(template_display_h * template_aspect)
                    template_display = cv2.resize(template_img, (template_display_w, template_display_h))
                    
                    # Place in top right corner
                    corner_x = w - template_display_w - 20
                    corner_y = 20
                    
                    # Add semi-transparent background behind template
                    template_bg = display_frame[corner_y:corner_y+template_display_h, corner_x:corner_x+template_display_w].copy()
                    cv2.rectangle(template_bg, (0, 0), (template_display_w, template_display_h), (0, 0, 0), -1)
                    display_frame[corner_y:corner_y+template_display_h, corner_x:corner_x+template_display_w] = cv2.addWeighted(
                        template_bg, 0.3, template_display, 0.7, 0)
                    
                    # Draw border around template
                    cv2.rectangle(display_frame, (corner_x-2, corner_y-2), 
                                 (corner_x+template_display_w+2, corner_y+template_display_h+2), (255, 255, 0), 3)
                    
                    # Show position number overlay on template
                    pos_overlay = f"Pos {sample_idx + 1}"
                    cv2.putText(display_frame, pos_overlay, (corner_x + 10, corner_y + 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)
                    cv2.putText(display_frame, pos_overlay, (corner_x + 10, corner_y + 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 2)
                
                # Show detected warp in bottom right corner if available
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
                    
                    # Use last detected warp or current frame
                    warp_to_save = last_warp if last_warp is not None else frame
                    
                    if warp_to_save is not None:
                        out_file = OUT_DIR / f"{current_card}{next_num}.png"
                        normalized = normalize(warp_to_save)
                        cv2.imwrite(str(out_file), normalized)
                        print(f"    ✓ Saved: {out_file.name}")
                        play_chime(capture=True)  # Play capture chime
                        sample_idx += 1  # Move to next sample
                        break  # Exit inner loop to show next sample
                    else:
                        print(f"    ⚠ No card detected - nothing saved. Try again.")
                
                # Check for keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("\nQuitting...")
                    card_idx = len(all_cards)  # Exit outer loop
                    sample_idx = SAMPLES_PER_CARD  # Exit sample loop
                    break
        
        # Only advance to next card if we completed all samples
        if sample_idx >= SAMPLES_PER_CARD:
            print(f"  ✓ Completed all {SAMPLES_PER_CARD} samples for {current_card}")
            card_idx += 1
        else:
            # User quit, exit
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone! Templates saved to: {OUT_DIR}")
    print(f"Completed {card_idx} out of 52 cards")

if __name__ == "__main__":
    main()
