import numpy as np, cv2
from guided_curve_tracker import trace_guided_curve

H, W = 1200, 500
img = np.full((H, W, 3), 245, np.uint8)

# gridlines
for y in range(0, H, 60):
    cv2.line(img, (0, y), (W, y), (170, 170, 170), 1)
for x in range(0, W, 50):
    cv2.line(img, (x, 0), (x, H), (170, 170, 170), 1)

ys = np.arange(H)
xa = (250 + 120*np.sin(ys/90.0) + 30*np.sin(ys/23.0)).astype(int)
xb = (250 + 140*np.cos(ys/110.0)).astype(int)

for y in ys[:-1]:
    if (y // 12) % 2 == 0:
        cv2.line(img, (xb[y], y), (xb[y+1], y+1), (20, 20, 20), 2)
    cv2.line(img, (xa[y], y), (xa[y+1], y+1), (20, 20, 20), 2)

anchor_rows = [30, 250, 500, 750, 1000, 1170]
anchors = [(int(xb[r]), int(r)) for r in anchor_rows]

res = trace_guided_curve(img, anchors, curve_style="dashed")
pts = np.array(res["points"])
for x, y in pts:
    e = abs(x - xb[y])
    if e > 20:
        print(f"y={y}, traced_x={x}, true_x={xb[y]}, error={e}")

