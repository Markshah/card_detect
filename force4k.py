import cv2

# Force Kiyo into 4K/24 MJPG mode – this is the only combo macOS reliably accepts first time
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not opened – trying index 1")
    cap = cv2.VideoCapture(1)

# THE magic line combo that makes macOS give us real 4K
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
cap.set(cv2.CAP_PROP_FPS, 24)

# Verify it actually took
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Resolution locked: {int(w)}×{int(h)}")

# Tiny delay so camera settles
cv2.waitKey(500)

while True:
    ret, frame = cap.read()
    if ret:
        # Resize for your screen so you can actually see it
        display = cv2.resize(frame, (1536, 864))   # 4K → nice big 864p window
        cv2.putText(display, "KIYO PRO ULTRA 4K – TWIST THE RING!", (30, 80),
                    cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 4)
        cv2.putText(display, "Press Q to quit", (30, 150),
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 255), 3)
        cv2.imshow("KIYO PRO ULTRA – REAL 4K FEED", display)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:   # q or Esc
        break

cap.release()
cv2.destroyAllWindows()
