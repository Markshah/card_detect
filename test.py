import cv2
import numpy as np

# List all video devices so we know the right index
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Device {i}: {int(w)}×{int(h)}")
        cap.release()

# Now open the Kiyo at full 4K (it’s almost always index 0 or 1)
cap = cv2.VideoCapture(0) # change to 1 or 2 if 0 doesn’t work
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
cap.set(cv2.CAP_PROP_FPS, 24)

while True:
    ret, frame = cap.read()
    if ret:
        # Show a resized version so it fits on your screen
        small = cv2.resize(frame, (1280, 720))
        cv2.putText(small, "REAL 4K FEED - Press Q to quit", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        cv2.imshow("Kiyo Pro Ultra - ACTUAL quality", small)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
