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

curve_lab = np.array([28.0, 128.0, 128.0], dtype=np.float32)

# manually trace inside _build_cost_map
bgr_crop = img
lab = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2LAB).astype(np.float32)
diff = np.abs(lab - curve_lab)
dist = np.sqrt(np.sum(diff**2, axis=2))
cost = np.clip(dist / 60.0, 0.0, 1.0)
print("After init:", cost.min(), cost.max())

gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
all_ink = cv2.adaptiveThreshold(
    gray, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10
).astype(np.uint8)

vgrid = np.zeros_like(cost, dtype=bool)
hgrid = np.zeros_like(cost, dtype=bool)

# skip gridlines for a sec
print("Before grid_dilated:", cost.min(), cost.max())
grid = vgrid | hgrid
grid_dilated = cv2.dilate(grid.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))) > 0
cost = np.where(grid_dilated, np.maximum(cost, 1.00), cost)
print("After grid_dilated:", cost.min(), cost.max())

curve_mask = (cost < 0.25).astype(np.uint8)
kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
closed_mask = cv2.morphologyEx(curve_mask, cv2.MORPH_CLOSE, kernel_v)
cost = np.where((closed_mask == 1) & (cost > 0.35), 0.35, cost)
print("Final cost:", cost.min(), cost.max())

