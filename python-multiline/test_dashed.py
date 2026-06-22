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
    # curve B dashed in the middle, solid elsewhere
    if not (400 < y < 700 and (y // 12) % 2 == 0):
        cv2.line(img, (xb[y], y), (xb[y+1], y+1), (20, 20, 20), 2)
    # curve A solid
    cv2.line(img, (xa[y], y), (xa[y+1], y+1), (20, 20, 20), 2)

anchor_rows = [30, 250, 500, 750, 1000, 1170]
anchors = [(int(xb[r]), int(r)) for r in anchor_rows]

res = trace_guided_curve(img, anchors, curve_style="dashed")
pts = np.array(res["points"])
err = []
for x, y in pts:
    err.append(abs(x - xb[y]))
err = np.array(err)
print("traced points:", len(pts))
print("mean |error| px:", round(err.mean(), 2))
print("max |error| px:", err.max())

vis = img.copy()
for x, y in pts:
    vis[y, max(0,x-1):x+2] = (0, 200, 0)
cv2.imwrite("test_dashed.png", vis)
