import numpy as np, cv2
dash_mask = np.zeros((100, 100), dtype=np.float32)
cv2.line(dash_mask, (20, 20), (30, 40), 1.0, 2)
cv2.line(dash_mask, (40, 60), (50, 80), 1.0, 2)

est_gap_px = 25.0
k_len = int(np.clip(est_gap_px * 1.5, 10, 80))
union_mask = np.zeros_like(dash_mask)
for angle in range(-60, 61, 15):
    kernel = np.zeros((k_len, k_len), dtype=np.uint8)
    cx, cy = k_len // 2, k_len // 2
    dx = int(cx * np.sin(np.radians(angle)))
    dy = int(cy * np.cos(np.radians(angle)))
    cv2.line(kernel, (cx - dx, cy - dy), (cx + dx, cy + dy), 1, 1)
    closed = cv2.morphologyEx(dash_mask, cv2.MORPH_CLOSE, kernel)
    union_mask = np.maximum(union_mask, closed)

# distance transform needs 8-bit single-channel image where 0 is background and >0 is foreground
# wait, cv2.distanceTransform measures distance to the nearest ZERO pixel!
# So we need union_mask == 1 to be 0 (so distance is 0), and background to be 1.
bg_mask = (1 - union_mask).astype(np.uint8)
dist_trans = cv2.distanceTransform(bg_mask, cv2.DIST_L2, 3)

reach = np.clip(1.0 - dist_trans / 15.0, 0.0, 1.0)
print(reach.shape, reach.max(), reach.min())
