import cv2, numpy as np
from guided_curve_tracker import trace_guided_curve, _build_cost_map, _sample_curve_appearance

img = np.full((1200, 500, 3), 255, dtype=np.uint8)
ys = np.arange(1200)
xa = (250 + 120*np.sin(ys/90.0) + 30*np.sin(ys/23.0)).astype(int)
xb = (250 + 140*np.cos(ys/110.0)).astype(int)

for y in ys[:-1]:
    if not (400 < y < 700 and (y // 12) % 2 == 0):
        cv2.line(img, (xb[y], y), (xb[y+1], y+1), (20, 20, 20), 2)
    cv2.line(img, (xa[y], y), (xa[y+1], y+1), (20, 20, 20), 2)

lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
curve_lab = _sample_curve_appearance(lab_img, [(xb[30], 30)])
cost, grid = _build_cost_map(img, curve_lab, is_dashed=True, suppress_gridlines=True, return_grid=True)

print("Gap region (y=400 to 415):")
for y in range(400, 416):
    c_xa = cost[y, xa[y]]
    c_xb = cost[y, xb[y]]
    print(f"y={y}: xa={xa[y]} (cost={c_xa:.2f}), xb={xb[y]} (cost={c_xb:.2f})")
