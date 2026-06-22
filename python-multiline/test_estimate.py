import numpy as np, cv2
from guided_curve_tracker import _sample_curve_appearance

H, W = 1000, 700
def build_log(dash_on=14, dash_off=12, amp=170, steep=False):
    img = np.full((H, W, 3), 248, np.uint8)
    for x in range(0, W, 35): cv2.line(img,(x,0),(x,H),(205,205,205),1)
    for y in range(0, H, 40): cv2.line(img,(0,y),(W,y),(205,205,205),1)
    ys = np.arange(H)
    a = 230 if steep else amp
    x_solid  = (350 + a*np.sin(ys/90.0)).astype(int)
    x_dashed = (350 - a*np.sin(ys/90.0)).astype(int)
    for y in ys[:-1]:
        cv2.line(img,(int(x_solid[y]),y),(int(x_solid[y+1]),y+1),(25,25,25),2)
    for y in ys[:-1]:
        if (y % (dash_on+dash_off)) < dash_on:
            cv2.line(img,(int(x_dashed[y]),y),(int(x_dashed[y+1]),y+1),(25,25,25),2)
    return img, x_solid, x_dashed

def _estimate_curve_style(bgr: np.ndarray, curve_lab: np.ndarray, anchors) -> tuple[bool, float]:
    if len(anchors) < 2:
        return False, 0.0
    
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_chroma = np.sqrt(curve_lab[1]**2 + curve_lab[2]**2)
    wL, wa, wb = (2.5, 1.0, 1.0) if target_chroma < 12 else (0.3, 2.5, 2.5)
    
    dL = lab[:, :, 0] - curve_lab[0]
    dL = np.where(dL < 0, dL * 0.8, dL)
    da = lab[:, :, 1] - curve_lab[1]
    db = lab[:, :, 2] - curve_lab[2]
    
    dist = np.sqrt(wL * dL * dL + wa * da * da + wb * db * db)
    cost = np.clip(dist / 60.0, 0.0, 1.0)
    
    ink_mask = (cost < 0.4).astype(np.uint8)
    
    all_gap_runs = []
    total_length = 0
    dashed_length = 0
    
    sorted_anchors = sorted(anchors, key=lambda p: p[1])
    
    for i in range(len(sorted_anchors) - 1):
        x0, y0 = sorted_anchors[i]
        x1, y1 = sorted_anchors[i+1]
        
        length = int(np.hypot(x1 - x0, y1 - y0))
        if length == 0: continue
        total_length += length
        
        line_mask = np.zeros_like(ink_mask)
        cv2.line(line_mask, (x0, y0), (x1, y1), 1, 5) # thickness 5
        
        ys, xs = np.where(line_mask == 1)
        if len(ys) == 0: continue
        
        order = np.argsort(ys)
        ys = ys[order]
        xs = xs[order]
        
        y_unique = np.unique(ys)
        vals = []
        for y in y_unique:
            row_xs = xs[ys == y]
            vals.append(np.max(ink_mask[y, row_xs]))
        vals = np.array(vals, dtype=int)
        
        padded = np.pad(vals, (1, 1), constant_values=1)
        diffs = np.diff(padded)
        gap_starts = np.where(diffs == -1)[0]
        gap_ends = np.where(diffs == 1)[0]
        
        gaps = gap_ends - gap_starts
        
        valid_gaps = gaps[gaps >= 3]
        if len(valid_gaps) > 0:
            all_gap_runs.extend(valid_gaps)
            if len(valid_gaps) >= 2:
                cv = np.std(valid_gaps) / (np.mean(valid_gaps) + 1e-6)
                if cv < 0.6 and np.median(valid_gaps) >= 4:
                    dashed_length += length
                    
    if len(all_gap_runs) == 0:
        return False, 0.0
        
    median_gap = float(np.median(all_gap_runs))
    is_dashed = (median_gap >= 4) and (dashed_length / max(1, total_length) >= 0.4)
    
    return is_dashed, median_gap if is_dashed else 0.0

img, x_solid, x_dashed = build_log()
anchor_rows = list(range(100, 900, 50))
anchors_dashed = [(int(x_dashed[r]), r) for r in anchor_rows]
anchors_solid = [(int(x_solid[r]), r) for r in anchor_rows]

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

print("Testing Dashed with thick line sampling:")
c_lab = _sample_curve_appearance(lab, anchors_dashed)
is_d, est_gap = _estimate_curve_style(img, c_lab, anchors_dashed)
print(f"is_dashed: {is_d}, gap: {est_gap}")

print("Testing Solid with thick line sampling:")
c_lab = _sample_curve_appearance(lab, anchors_solid)
is_d, est_gap = _estimate_curve_style(img, c_lab, anchors_solid)
print(f"is_dashed: {is_d}, gap: {est_gap}")
