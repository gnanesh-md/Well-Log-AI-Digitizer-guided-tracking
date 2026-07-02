import cv2
import numpy as np

H, W = 900, 420
img = np.full((H, W, 3), 245, np.uint8)

# Gridlines
for y in range(0, H, 60):
    cv2.line(img, (0, y), (W, y), (185, 185, 185), 1)
for x in range(0, W, 50):
    cv2.line(img, (x, 0), (x, H), (185, 185, 185), 1)

ys = np.arange(H)
xd = (210 + 70 * np.sin(ys / 70.0) + 18 * np.sin(ys / 19.0)).astype(int)
xs = xd - 18

for y in ys[:-1]:
    if (y // 16) % 2 == 0:
        cv2.line(img, (int(xd[y]), int(y)), (int(xd[y + 1]), int(y + 1)), (20, 20, 20), 2)
    cv2.line(img, (int(xs[y]), int(y)), (int(xs[y + 1]), int(y + 1)), (20, 20, 20), 2)

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
curve_lab = np.array([16., 128., 128.])

dL = lab[:, :, 0] - curve_lab[0]
dL = np.where(dL < 0, dL * 0.8, dL)
da = lab[:, :, 1] - curve_lab[1]
db = lab[:, :, 2] - curve_lab[2]
dist = np.sqrt(2.5 * dL * dL + 1.0 * da * da + 1.0 * db * db)
cost = np.clip(dist / 60.0, 0.0, 1.0)
ink = (cost < 0.35).astype(np.uint8)

for ksize in [(3, 7), (3, 5), (3, 3), (1, 5), (1, 3)]:
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    closed_ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, close_k)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed_ink, connectivity=8)
    
    # Check component containing xd[200] at y=200
    lbl_d = labels[200, int(xd[200])]
    lbl_s = labels[200, int(xs[200])]
    
    h_d = stats[lbl_d, cv2.CC_STAT_HEIGHT] if lbl_d > 0 else 0
    h_s = stats[lbl_s, cv2.CC_STAT_HEIGHT] if lbl_s > 0 else 0
    
    same = (lbl_d == lbl_s) if (lbl_d > 0 and lbl_s > 0) else False
    
    print(f"Kernel {ksize}: num_labels={num_labels}, same_label={same}, height_d={h_d}, height_s={h_s}")
