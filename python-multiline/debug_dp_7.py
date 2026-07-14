import numpy as np
import cv2
from guided_curve_tracker import trace_guided_curve
import guided_curve_tracker

H, W = 1000, 500
img = np.full((H, W, 3), (180, 205, 230), np.uint8)

anchors = [(400, 100), (400, 900)]

# Just test _build_cost_map directly!
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
curve_lab = np.array([28.0, 128.0, 128.0], dtype=np.float32)
cost, vgrid, hgrid = guided_curve_tracker._build_cost_map(img, curve_lab, False, True)

print(f"Cost min: {cost.min()}, max: {cost.max()}")
print(f"vgrid sum: {vgrid.sum()}")
print(f"hgrid sum: {hgrid.sum()}")
