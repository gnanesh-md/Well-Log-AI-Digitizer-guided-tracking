import numpy as np
import cv2
from guided_curve_tracker import trace_guided_curve

# Create a synthetic image
H, W = 1000, 500
img = np.full((H, W, 3), (180, 205, 230), np.uint8)      # paper

# Draw a black curve that drops down, spikes left, and drops down again
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
res = trace_guided_curve(img, anchors, curve_style="solid")

for p in res["points"]:
    if p[1] in (290, 310, 325, 340, 360, 410):
        print(f"y={p[1]}, x={p[0]}")

trace_min_x = min(p[0] for p in res["points"])
print(f"Trace min x: {trace_min_x}")
