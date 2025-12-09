#!/usr/bin/env python3
"""
Interactive ROI (Region of Interest) selector for card detection.
Click and drag to select the area where cards will be detected.
"""

import cv2
import os
import sys
import time

# Load camera index from env
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))

# ROI selection state
drawing = False
ix, iy = -1, -1
fx, fy = -1, -1
roi_coords = None
scale_x = 1.0  # Scale factors for coordinate conversion
scale_y = 1.0

# Performance optimization: frame skipping
FRAME_SKIP = 3  # Process every 3rd frame (reduces CPU by ~66%)
TARGET_FPS = 10  # Target display FPS

def mouse_callback(event, x, y, flags, param):
    """Handle mouse events for ROI selection"""
    global ix, iy, fx, fy, drawing, roi_coords, scale_x, scale_y
    
    # Convert display coordinates back to full resolution coordinates
    if scale_x > 0 and scale_y > 0:
        x_full = int(x / scale_x)
        y_full = int(y / scale_y)
    else:
        x_full, y_full = x, y
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x_full, y_full
        fx, fy = x_full, y_full
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            fx, fy = x_full, y_full
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        fx, fy = x_full, y_full
        # Calculate ROI: x, y, width, height (in full resolution coordinates)
        x1, y1 = min(ix, fx), min(iy, fy)
        x2, y2 = max(ix, fx), max(iy, fy)
        roi_coords = (x1, y1, x2 - x1, y2 - y1)

def main():
    global roi_coords
    
    print("=" * 60)
    print("ROI Setup Tool for Brio 4K Camera")
    print("=" * 60)
    print("\nInstructions:")
    print("1. Click and drag to select the card detection area")
    print("2. Press 's' to save the ROI to env file")
    print("3. Press 'r' to reset selection")
    print("4. Press 'q' or ESC to quit")
    print("=" * 60)
    
    # Open camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {CAMERA_INDEX}")
        print("Try setting CAMERA_INDEX=1 if you have multiple cameras")
        return
    
    try:
        # Configure for Brio 4K - MJPG codec works reliably on macOS
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
        cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS to reduce CPU
        
        # Get actual resolution
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"\nCamera resolution: {w}x{h}")
        print(f"Frame skip: {FRAME_SKIP} (processing every {FRAME_SKIP} frames)")
    except Exception as e:
        print(f"[WARNING] Could not set camera properties: {e}")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Using default resolution: {w}x{h}")
    
    # Create window and set mouse callback
    window_name = "ROI Selector - Click and drag to select area"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    # Load existing ROI from env if available
    env_file = os.path.join(os.path.dirname(__file__), "env")
    existing_roi = "0,0,0,0"
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                if line.strip().startswith("ROI="):
                    existing_roi = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if existing_roi != "0,0,0,0":
                        parts = existing_roi.split(",")
                        if len(parts) == 4:
                            roi_coords = tuple(int(p.strip()) for p in parts)
                            print(f"\nLoaded existing ROI: {existing_roi}")
                    break
    
    print("\nStarting camera feed...")
    
    # Performance: limit display size to reduce resize overhead
    MAX_DISPLAY_WIDTH = 1280  # Max display width
    MAX_DISPLAY_HEIGHT = 720  # Max display height
    
    frame_count = 0
    last_frame_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame")
            break
        
        frame_count += 1
        
        # Skip frames to reduce CPU usage, but process more frequently when dragging
        skip_factor = 1 if drawing else FRAME_SKIP
        if frame_count % skip_factor != 0:
            # Still need to read frames to keep buffer clear, but don't process
            continue
        
        # Throttle display updates
        current_time = time.time()
        elapsed = current_time - last_frame_time
        if elapsed < (1.0 / TARGET_FPS):
            time.sleep((1.0 / TARGET_FPS) - elapsed)
        last_frame_time = time.time()
        
        # Create display frame (resize for screen - more aggressive downscaling)
        display_scale = min(MAX_DISPLAY_WIDTH / w, MAX_DISPLAY_HEIGHT / h, 1.0)
        display_w = int(w * display_scale)
        display_h = int(h * display_scale)
        
        # Use INTER_AREA for better downscaling performance
        display_frame = cv2.resize(frame, (display_w, display_h), interpolation=cv2.INTER_AREA)
        
        # Scale coordinates for display (make global for mouse callback)
        global scale_x, scale_y
        scale_x = display_w / w
        scale_y = display_h / h
        
        # Draw current selection
        if drawing or roi_coords:
            if drawing:
                # Draw while dragging (coordinates already in full res, scale for display)
                x1, y1 = int(ix * scale_x), int(iy * scale_y)
                x2, y2 = int(fx * scale_x), int(fy * scale_y)
            else:
                # Draw saved ROI (coordinates already in full res, scale for display)
                x, y, w_roi, h_roi = roi_coords
                x1, y1 = int(x * scale_x), int(y * scale_y)
                x2, y2 = int((x + w_roi) * scale_x), int((y + h_roi) * scale_y)
            
            # Draw rectangle
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw corner markers
            corner_size = 20
            cv2.line(display_frame, (x1, y1), (x1 + corner_size, y1), (0, 255, 0), 3)
            cv2.line(display_frame, (x1, y1), (x1, y1 + corner_size), (0, 255, 0), 3)
            cv2.line(display_frame, (x2, y1), (x2 - corner_size, y1), (0, 255, 0), 3)
            cv2.line(display_frame, (x2, y1), (x2, y1 + corner_size), (0, 255, 0), 3)
            cv2.line(display_frame, (x1, y2), (x1 + corner_size, y2), (0, 255, 0), 3)
            cv2.line(display_frame, (x1, y2), (x1, y2 - corner_size), (0, 255, 0), 3)
            cv2.line(display_frame, (x2, y2), (x2 - corner_size, y2), (0, 255, 0), 3)
            cv2.line(display_frame, (x2, y2), (x2, y2 - corner_size), (0, 255, 0), 3)
            
            # Display ROI coordinates
            if roi_coords:
                x, y, w_roi, h_roi = roi_coords
                coord_text = f"ROI: {x},{y},{w_roi},{h_roi}"
                cv2.putText(display_frame, coord_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Add instructions overlay
        cv2.putText(display_frame, "Click & drag to select ROI | 's'=save | 'r'=reset | 'q'=quit",
                   (10, display_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow(window_name, display_frame)
        
        # Use waitKey with longer delay to reduce CPU polling
        key = cv2.waitKey(30) & 0xFF  # 30ms = ~33 FPS max, but we're throttling anyway
        if key == ord('q') or key == 27:  # q or ESC
            break
        elif key == ord('s'):  # Save
            if roi_coords:
                x, y, w_roi, h_roi = roi_coords
                roi_str = f"{x},{y},{w_roi},{h_roi}"
                
                # Update env file
                if os.path.exists(env_file):
                    with open(env_file, 'r') as f:
                        lines = f.readlines()
                    
                    updated = False
                    with open(env_file, 'w') as f:
                        for line in lines:
                            if line.strip().startswith("ROI="):
                                f.write(f'ROI="{roi_str}"\n')
                                updated = True
                            else:
                                f.write(line)
                    
                    if not updated:
                        # Add ROI line if it doesn't exist
                        with open(env_file, 'a') as f:
                            f.write(f'\nROI="{roi_str}"\n')
                    
                    print(f"\n[SUCCESS] ROI saved to env file: {roi_str}")
                    print(f"ROI covers {w_roi}x{h_roi} pixels at position ({x}, {y})")
                else:
                    print(f"[ERROR] env file not found at {env_file}")
            else:
                print("[WARNING] No ROI selected. Please select an area first.")
        elif key == ord('r'):  # Reset
            roi_coords = None
            print("[INFO] ROI selection reset")
    
    cap.release()
    cv2.destroyAllWindows()
    
    if roi_coords:
        x, y, w_roi, h_roi = roi_coords
        print(f"\nFinal ROI: {x},{y},{w_roi},{h_roi}")
        print("(Press 's' to save before quitting next time)")

if __name__ == "__main__":
    main()

