#!/usr/bin/env python3
"""
Simple tool to capture deck back templates.
Place a face-down card in view and click to capture.
"""

import cv2
import os
import numpy as np
import time
from dotenv import load_dotenv

# Load env
if os.path.exists("env"):
    load_dotenv("env")

CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
ROI = tuple(int(p or "0") for p in os.getenv("ROI", "0,0,0,0").split(",")[:4]) if os.getenv("ROI") else (0,0,0,0)
WARP_W = int(os.getenv("WARP_W", "500"))
WARP_H = int(os.getenv("WARP_H", "700"))

TEMPL_DIR = os.getenv("CARD_FULL_TEMPL_DIR", "./templates")
os.makedirs(TEMPL_DIR, exist_ok=True)

def _roi(frame, r):
    x, y, w, h = r
    if w <= 0 or h <= 0:
        return frame, 0, 0
    x2, y2 = min(frame.shape[1], x+w), min(frame.shape[0], y+h)
    return frame[y:y2, x:x2].copy(), x, y

def _normalize_L(gray):
    """Normalize lightness channel for better edge detection."""
    mean = gray.mean()
    std = gray.std() + 1e-6
    normalized = ((gray.astype(np.float32) - mean) / std * 50 + 128).clip(0, 255).astype(np.uint8)
    return normalized

def _warp_from_contour(img_bgr, cnt, out_size=(WARP_W, WARP_H)):
    """Extract card from contour using perspective transform."""
    box = cv2.minAreaRect(cnt)
    box_pts = cv2.boxPoints(box)
    box_pts = np.float32(box_pts)
    
    # Order points: top-left, top-right, bottom-right, bottom-left
    s = box_pts.sum(axis=1)
    diff = np.diff(box_pts, axis=1)
    tl = box_pts[np.argmin(s)]
    br = box_pts[np.argmax(s)]
    tr = box_pts[np.argmin(diff)]
    bl = box_pts[np.argmax(diff)]
    
    src = np.float32([tl, tr, br, bl])
    W, H = out_size
    dst = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    
    M = cv2.getPerspectiveTransform(src, dst)
    warp = cv2.warpPerspective(img_bgr, M, (W, H))
    return warp, box_pts.astype(np.int32), (W, H)

def _card_candidates(mask, frame_w, frame_h):
    """Find card-like contours."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 3000:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if min(w, h) < 85:
            continue
        hull = cv2.convexHull(cnt)
        solidity = area / (cv2.contourArea(hull) + 1e-6)
        (cx, cy), (mw, mh), ang = cv2.minAreaRect(cnt)
        asp = max(mw, mh) / max(1.0, min(mw, mh))
        if not (1.25 <= asp <= 1.75):
            continue
        if solidity < 0.88:
            continue
        yield cnt

capture_state = {'capture': False}

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        capture_state['capture'] = True

def main():
    # Define your deck types with display colors
    DECK_TYPES = [
        ("RED_WHITE", (0, 0, 255)),      # Red/white deck - Red color
        ("BLUE_WHITE", (255, 0, 0)),    # Blue/white deck - Blue color
        ("PINK_YELLOW", (203, 192, 255)), # Pink/yellow deck - Pink color
        ("LIGHT_BLUE_YELLOW", (255, 255, 0))  # Light blue/yellow deck - Cyan color
    ]
    
    # Sample positions/rotations (similar to regular card template capture)
    SAMPLE_POSITIONS = [
        "Center, normal orientation",
        "Left position",
        "Right position",
        "Up position",
        "Down position",
        "Center, rotated ~15°",
        "Center, rotated ~45°"
    ]
    
    SAMPLES_PER_DECK = len(SAMPLE_POSITIONS)  # Number of samples to capture per deck
    
    print("=" * 60)
    print("Deck Back Template Capture")
    print("=" * 60)
    print(f"\nWill capture {SAMPLES_PER_DECK} samples for each deck:")
    for i, (deck, _) in enumerate(DECK_TYPES, 1):
        print(f"  {i}. {deck.replace('_', '/')}")
    print("\nInstructions:")
    print("  - Click mouse to capture current sample")
    print("  - Script will automatically advance to next deck after all samples")
    print("  - Press 'n' to skip to next deck early")
    print("  - Press 'q' to quit")
    print("=" * 60)
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {CAMERA_INDEX}")
        return
    
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
        cap.set(cv2.CAP_PROP_FPS, 24)
    except Exception:
        pass
    
    window_name = "Deck Back Capture - Click to capture, N for next deck, Q to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    deck_idx = 0
    sample_num = 1
    samples_captured = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        proc, offx, offy = _roi(frame, ROI) if sum(ROI) != 0 else (frame, 0, 0)
        gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        norm = _normalize_L(gray)
        _, th = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        detected_warp = None
        display_frame = frame.copy()
        
        for cnt in _card_candidates(th, proc.shape[1], proc.shape[0]):
            warp, box, (mw, mh) = _warp_from_contour(proc, cnt)
            detected_warp = warp
            box_int = box.astype(np.int32)
            cv2.drawContours(display_frame, [box_int + [offx, offy]], -1, (0, 255, 0), 2)
            break
        
        # Show detected warp preview
        if detected_warp is not None:
            h, w = display_frame.shape[:2]
            warp_display = cv2.resize(detected_warp, (200, 280))
            h_w, w_w = warp_display.shape[:2]
            warp_x = w - w_w - 20
            warp_y = h - h_w - 20
            display_frame[warp_y:warp_y+h_w, warp_x:warp_x+w_w] = warp_display
            cv2.rectangle(display_frame, (warp_x-2, warp_y-2), (warp_x+w_w+2, warp_y+h_w+2), (0, 255, 0), 3)
        
        # Get current deck info
        if deck_idx < len(DECK_TYPES):
            current_deck, deck_color = DECK_TYPES[deck_idx]
            deck_display_name = current_deck.replace('_', '/')
        else:
            current_deck = None
            deck_display_name = "ALL COMPLETE"
            deck_color = (0, 255, 0)  # Green when done
        
        current_position = SAMPLE_POSITIONS[samples_captured] if samples_captured < len(SAMPLE_POSITIONS) else "Done"
        
        # Large, prominent deck color indicator at top
        h, w = display_frame.shape[:2]
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), deck_color, -1)
        cv2.addWeighted(overlay, 0.3, display_frame, 0.7, 0, display_frame)
        
        # Large deck name text
        deck_text = f"CAPTURING: {deck_display_name}"
        text_size = cv2.getTextSize(deck_text, cv2.FONT_HERSHEY_DUPLEX, 1.5, 4)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(display_frame, deck_text,
                   (text_x, 50), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 4)
        
        # Sample progress
        progress_text = f"Sample {samples_captured + 1}/{SAMPLES_PER_DECK}"
        progress_size = cv2.getTextSize(progress_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        progress_x = (w - progress_size[0]) // 2
        cv2.putText(display_frame, progress_text,
                   (progress_x, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Position text
        position_text = f"Position: {current_position}"
        cv2.putText(display_frame, position_text,
                   (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        instruction_text = "Click to capture | N=skip to next deck | Q=quit"
        cv2.putText(display_frame, instruction_text,
                   (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show progress for all decks
        progress_y = 200
        cv2.putText(display_frame, "Deck Progress:",
                   (10, progress_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        progress_y += 30
        
        for i, (deck, deck_color_bgr) in enumerate(DECK_TYPES):
            if i == deck_idx:
                # Current deck - highlight with its color
                prefix = "→ "
                text_color = (255, 255, 255)  # White text on colored background
                bg_color = deck_color_bgr
            elif i < deck_idx:
                # Completed deck - green checkmark
                prefix = "✓ "
                text_color = (0, 255, 0)
                bg_color = None
            else:
                # Upcoming deck - gray
                prefix = "  "
                text_color = (128, 128, 128)
                bg_color = None
            
            deck_name = deck.replace('_', '/')
            # Count existing samples for this deck
            existing = 0
            if os.path.exists(TEMPL_DIR):
                existing = len([f for f in os.listdir(TEMPL_DIR) 
                               if f.startswith(f"BACK_{deck}") and f.endswith(('.png', '.jpg', '.jpeg'))])
            
            progress_text = f"{prefix}{deck_name}: {existing}/{SAMPLES_PER_DECK} samples"
            
            # Draw background for current deck
            if bg_color is not None:
                text_size = cv2.getTextSize(progress_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(display_frame, (5, progress_y - 20), 
                             (text_size[0] + 15, progress_y + 5), bg_color, -1)
            
            cv2.putText(display_frame, progress_text,
                       (10, progress_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            progress_y += 25
        
        cv2.imshow(window_name, display_frame)
        
        if capture_state['capture']:
            capture_state['capture'] = False
            if current_deck is None:
                print("\n[INFO] All decks completed!")
                continue
            
            if detected_warp is not None:
                filename = f"BACK_{current_deck}{sample_num}.png"
                filepath = os.path.join(TEMPL_DIR, filename)
                cv2.imwrite(filepath, detected_warp)
                position_name = SAMPLE_POSITIONS[samples_captured] if samples_captured < len(SAMPLE_POSITIONS) else "sample"
                print(f"  ✓ Saved: {filename} ({samples_captured + 1}/{SAMPLES_PER_DECK} - {position_name})")
                sample_num += 1
                samples_captured += 1
                
                # Show next position hint
                if samples_captured < SAMPLES_PER_DECK:
                    next_position = SAMPLE_POSITIONS[samples_captured]
                    print(f"  → Next: {next_position}")
                
                # Auto-advance to next deck if we've captured enough samples
                if samples_captured >= SAMPLES_PER_DECK:
                    print(f"\n✓ Completed {deck_display_name} ({SAMPLES_PER_DECK} samples)")
                    deck_idx += 1
                    samples_captured = 0
                    sample_num = 1  # Reset sample number for next deck
                    if deck_idx >= len(DECK_TYPES):
                        print("\n🎉 All decks completed!")
                        # Don't break immediately - let user see completion message
                        time.sleep(2)
                        break
                    else:
                        next_deck, next_color = DECK_TYPES[deck_idx]
                        next_deck_name = next_deck.replace('_', '/')
                        print(f"\n{'='*60}")
                        print(f"→ AUTOMATICALLY MOVING TO: {next_deck_name}")
                        print(f"{'='*60}")
                        print(f"  First position: {SAMPLE_POSITIONS[0]}")
                        print(f"  Place a {next_deck_name} deck card in view")
            else:
                print("  [WARNING] No card detected. Try again.")
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('n'):
            # Manually advance to next deck
            if current_deck:
                remaining = SAMPLES_PER_DECK - samples_captured
                print(f"\n→ Skipping {remaining} remaining sample(s) for {deck_display_name}")
            deck_idx += 1
            samples_captured = 0
            sample_num = 1  # Reset sample number for next deck
            if deck_idx >= len(DECK_TYPES):
                print("\n🎉 All decks completed!")
                break
            else:
                next_deck, next_color = DECK_TYPES[deck_idx]
                next_deck_name = next_deck.replace('_', '/')
                print(f"\n{'='*60}")
                print(f"→ MOVING TO: {next_deck_name}")
                print(f"{'='*60}")
                print(f"  First position: {SAMPLE_POSITIONS[0]}")
                print(f"  Place a {next_deck_name} deck card in view")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Count total samples captured
    total_samples = 0
    print("\n" + "=" * 60)
    print("Capture Summary:")
    print("=" * 60)
    if os.path.exists(TEMPL_DIR):
        for deck_name, _ in DECK_TYPES:
            deck_files = [f for f in os.listdir(TEMPL_DIR) 
                         if f.startswith(f"BACK_{deck_name}") and f.endswith(('.png', '.jpg', '.jpeg'))]
            count = len(deck_files)
            total_samples += count
            display_name = deck_name.replace('_', '/')
            status = "✓" if count > 0 else "✗"
            print(f"  {status} {display_name}: {count} sample(s)")
    print(f"\nTotal: {total_samples} deck back template(s)")
    print("\nTo enable deck back template matching, add to env file:")
    print("  USE_DECK_BACK_TEMPLATES=1")
    print("  DECK_BACK_TEMPLATE_THRESH=0.50")

if __name__ == "__main__":
    main()

