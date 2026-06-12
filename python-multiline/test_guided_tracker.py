"""Validate guided tracing on a synthetic well log: two crossing black curves,
a dashed segment, and gridlines. Anchors are placed on curve A only; we then
measure how far the traced path deviates from ground-truth curve A."""
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
# ground truth curve A (black, wiggly) and curve B (black, crosses A several times)
xa = (250 + 120*np.sin(ys/90.0) + 30*np.sin(ys/23.0)).astype(int)
xb = (250 + 140*np.cos(ys/110.0)).astype(int)

for y in ys[:-1]:
    cv2.line(img, (xa[y], y), (xa[y+1], y+1), (20, 20, 20), 2)
    # curve B dashed in the middle, solid elsewhere
    if not (400 < y < 700 and (y // 12) % 2 == 0):
        cv2.line(img, (xb[y], y), (xb[y+1], y+1), (20, 20, 20), 2)

# add noise specks
rng = np.random.default_rng(0)
for _ in range(800):
    x, y = rng.integers(0, W), rng.integers(0, H)
    img[y, x] = (60, 60, 60)

# user clicks ~6 rough anchors on curve A (with up to 6 px click error)
anchor_rows = [30, 250, 500, 750, 1000, 1170]
anchors = [(int(xa[r] + rng.integers(-6, 7)), int(r + rng.integers(-4, 5))) for r in anchor_rows]

res = trace_guided_curve(img, anchors)
pts = np.array(res["points"])
err = []
for x, y in pts:
    err.append(abs(x - xa[y]))
err = np.array(err)
print("traced points:", len(pts))
print("mean |error| px:", round(err.mean(), 2))
print("median |error| px:", round(np.median(err), 2))
print("p95 |error| px:", round(np.percentile(err, 95), 2))
print("max |error| px:", err.max())
print("within 2px:", round((err <= 2).mean()*100, 1), "%")
print("confidence:", res["confidence"])

# render result for visual check
vis = img.copy()
for x, y in pts:
    vis[y, max(0,x-1):x+2] = (0, 200, 0)
for x, y in res["snapped_anchors"]:
    cv2.circle(vis, (x, y), 5, (0, 0, 255), -1)
cv2.imwrite("/home/claude/test_trace.png", vis)
print("saved /home/claude/test_trace.png")
