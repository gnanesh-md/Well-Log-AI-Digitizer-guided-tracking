import numpy as np
import cv2
from guided_curve_tracker import trace_guided_curve
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

anchors = [(400, 100), (400, 900)]

orig_build = guided_curve_tracker._build_cost_map
def fake_build(*args, **kwargs):
    print(f"curve_lab sampled: {args[1]}")
    cost, vgrid, hgrid = orig_build(*args, **kwargs)
    print(f"Cost min: {cost.min()}, max: {cost.max()}")
    print(f"Cost at (100, 350): {cost[350, 100]}")
    return cost, vgrid, hgrid
guided_curve_tracker._build_cost_map = fake_build

res = trace_guided_curve(img, anchors, curve_style="solid")
