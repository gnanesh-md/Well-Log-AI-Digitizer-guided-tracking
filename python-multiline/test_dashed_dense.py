import cv2, numpy as np
from guided_curve_tracker import trace_guided_curve

img = cv2.imread("test_dashed.png")
# Generate same xb to get anchors
h, w = img.shape[:2]
ys = np.arange(h)
xb = 250 + 140 * np.cos(ys / 110.0)

# DENSE ANCHORS!
anchor_rows = [30, 150, 250, 375, 500, 625, 750, 875, 1000, 1170]
anchors = [(int(xb[r]), int(r)) for r in anchor_rows]

res = trace_guided_curve(img, anchors, curve_style="dashed")
pts = np.array(res["path"])

# calculate error
error = []
for p in pts:
    x, y = p
    true_x = xb[y]
    error.append(abs(x - true_x))

print(f"traced points: {len(pts)}")
print(f"mean |error| px: {np.mean(error):.2f}")
print(f"max |error| px: {np.max(error)}")
