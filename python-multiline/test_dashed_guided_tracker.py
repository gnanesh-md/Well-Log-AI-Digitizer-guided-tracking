"""Regression check for human-guided dashed curve tracking."""

import cv2
import numpy as np

from guided_curve_tracker import trace_guided_curve


H, W = 900, 420
img = np.full((H, W, 3), 245, np.uint8)

for y in range(0, H, 60):
    cv2.line(img, (0, y), (W, y), (185, 185, 185), 1)
for x in range(0, W, 50):
    cv2.line(img, (x, 0), (x, H), (185, 185, 185), 1)

ys = np.arange(H)
xd = (210 + 70 * np.sin(ys / 70.0) + 18 * np.sin(ys / 19.0)).astype(int)
xs = (230 + 78 * np.sin(ys / 70.0) + 18 * np.sin(ys / 19.0)).astype(int)

for y in ys[:-1]:
    if (y // 16) % 2 == 0:
        cv2.line(img, (int(xd[y]), int(y)), (int(xd[y + 1]), int(y + 1)), (20, 20, 20), 2)
    cv2.line(img, (int(xs[y]), int(y)), (int(xs[y + 1]), int(y + 1)), (20, 20, 20), 2)

rng = np.random.default_rng(4)
anchor_rows = [35, 160, 290, 430, 570, 710, 860]
anchors = [
    (int(xd[r] + rng.integers(-4, 5)), int(r + rng.integers(-3, 4)))
    for r in anchor_rows
]

res = trace_guided_curve(
    img,
    anchors,
    curve_style="dashed",
    corridor_pad=140,
    smooth_window=1,
)
pts = np.array(res["points"])
err = np.array([abs(x - int(xd[y])) for x, y in pts if 0 <= y < H])

print("traced points:", len(pts))
print("mean |error| px:", round(float(err.mean()), 2))
print("median |error| px:", round(float(np.median(err)), 2))
print("p95 |error| px:", round(float(np.percentile(err, 95)), 2))
print("max |error| px:", int(err.max()))
print("within 4px:", round(float((err <= 4).mean() * 100), 1), "%")
print("confidence:", res["confidence"])
print("anchors:", anchors)
print("snapped anchors:", res["snapped_anchors"])

vis = img.copy()
for x, y in pts:
    vis[y, max(0, x - 1):x + 2] = (0, 200, 0)
for x, y in res["snapped_anchors"]:
    cv2.circle(vis, (x, y), 5, (0, 0, 255), -1)
cv2.imwrite("test_dashed_guided_trace.png", vis)
print("saved test_dashed_guided_trace.png")

if np.median(err) > 4 or np.percentile(err, 95) > 30:
    raise SystemExit("dashed tracker deviated too far from dashed ground truth")
