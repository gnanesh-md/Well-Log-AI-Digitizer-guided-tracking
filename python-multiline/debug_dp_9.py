import numpy as np
import cv2
import guided_curve_tracker

H, W = 1000, 500
img = np.full((H, W, 3), (180, 205, 230), np.uint8)

pts = []
for y in range(0, 300):
    pts.append((400, y))
for y in range(300, 350):
    x = 400 - int((y - 300) / 50.0 * 300)
    pts.append((x, y))
for y in range(350, 400):
    x = 100 + int((y - 350) / 50.0 * 300)
    pts.append((x, y))
for y in range(400, 1000):
    pts.append((400, y))

for i in range(len(pts) - 1):
    cv2.line(img, pts[i], pts[i + 1], (30, 30, 30), 2)

print("Min BGR:", np.min(img, axis=(0,1)))

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
curve_lab = np.array([28.0, 128.0, 128.0], dtype=np.float32)
diff = np.abs(lab - curve_lab)
dist = np.sqrt(np.sum(diff**2, axis=2))
cost = np.clip(dist / 60.0, 0.0, 1.0)
print(f"dist min: {dist.min()}")
print(f"cost min: {cost.min()}")

