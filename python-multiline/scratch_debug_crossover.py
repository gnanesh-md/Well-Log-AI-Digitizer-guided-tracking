import cv2
import numpy as np

H, W = 1200, 500
img = np.full((H, W, 3), 245, dtype=np.uint8)

# Gridlines
for y in range(0, H, 60):
    cv2.line(img, (0, y), (W, y), (170, 170, 170), 1)
for x in range(0, W, 50):
    cv2.line(img, (x, 0), (x, H), (170, 170, 170), 1)

ys = np.arange(H)
xa = (250 + 120 * np.sin(ys / 90.0) + 30 * np.sin(ys / 23.0)).astype(int)
xb = (250 + 140 * np.cos(ys / 110.0)).astype(int)

for y in ys[:-1]:
    if (y // 12) % 2 == 0:
        cv2.line(img, (xb[y], y), (xb[y+1], y+1), (20, 20, 20), 2)
    cv2.line(img, (xa[y], y), (xa[y+1], y+1), (20, 20, 20), 2)

anchor_rows = [30, 250, 500, 750, 1000, 1170]
anchors = [(int(xb[r]), int(r)) for r in anchor_rows]

import guided_curve_tracker

# Let's write the fully corrected version of `_trace_all_segments`
def patched_trace_all_segments(
    base_cost: np.ndarray,
    snapped: list,
    is_dashed: bool,
    max_slope_px: int,
    move_penalty: float,
    curvature_penalty: float,
    corridor_pad: int,
    x_min: int,
    x_max: int,
    prior_trajectory=None,
    exclusive=False,
    est_gap_px=15.0,
    is_dotted=False,
):
    all_points = []
    segments = []
    prev_exit_slope = -999.0
    
    prior_map = None
    if prior_trajectory is not None:
        prior_map = {pt[1]: pt[0] for pt in prior_trajectory if pt is not None}
        
    for i in range(len(snapped) - 1):
        p_start = snapped[i]
        p_end = snapped[i + 1]
        (xs_s, ys_s) = p_start
        (xe_s, ye_s) = p_end
        dy = abs(ye_s - ys_s)
        dx = abs(xe_s - xs_s)
        
        track_w = x_max - x_min if x_max > x_min else 800
        is_wrap_jump = False # dashed is always False
        
        if dy > 0:
            avg_slope = dx / dy
            max_allowed_slope = max(120, int(track_w + 50))
            seg_max_slope = int(min(max_allowed_slope, max(max_slope_px, avg_slope * 1.2 + 2)))
            if is_dashed:
                seg_curvature_penalty = float(np.clip(curvature_penalty - avg_slope * 0.02, 0.03, curvature_penalty))
            else:
                seg_curvature_penalty = float(np.clip(curvature_penalty - avg_slope * 0.05, 0.01, curvature_penalty))
        else:
            seg_max_slope = max_slope_px
            seg_curvature_penalty = curvature_penalty * 0.75 if is_dashed else curvature_penalty * 0.25

        if is_dashed:
            seg_prior_x = None
            if prior_map is not None:
                seg_prior_x = np.array([prior_map.get(y, xs_s + (y - ys_s)/(dy if dy > 0 else 1)*(xe_s - xs_s)) for y in range(ys_s, ye_s + 1)])
            
            # Use the patched gap fill with corridor_pad instead of min(corridor_pad, 60)
            seg_cost_map = patched_gap_fill(
                base_cost, p_start, p_end, x_min=x_min, x_max=x_max,
                prior_x=seg_prior_x, exclusive=exclusive, est_gap_px=est_gap_px,
                is_dotted=is_dotted,
                corridor_half_width=corridor_pad,
                bridge_half_width=4
            )
            # Correct guide weight and trajectory!
            seg_guide_weight = 0.0002 if prior_map is not None else 0.0
            seg_trajectory = seg_prior_x if prior_map is not None else None
        else:
            seg_cost_map = base_cost
            seg_guide_weight = 0.0
            seg_trajectory = None

        seg_pts, seg_cost, worst_pt = guided_curve_tracker._trace_segment_dp(
            seg_cost_map, p_start, p_end,
            max_slope_px=seg_max_slope,
            move_penalty=move_penalty,
            curvature_penalty=seg_curvature_penalty,
            corridor_pad=int(min(corridor_pad, dx + 40)) if dy > 0 else corridor_pad,
            start_slope=prev_exit_slope if prev_exit_slope != -999.0 else None,
            endpoint_slope_weight=0.04 if prev_exit_slope != -999.0 else 0.0,
            x_min=x_min,
            x_max=x_max,
            guide_weight=seg_guide_weight,
            trajectory=seg_trajectory,
        )
        
        if len(seg_pts) >= 8:
            pts_end = seg_pts[-8:]
            if pts_end[-1][1] != pts_end[0][1]:
                prev_exit_slope = (pts_end[-1][0] - pts_end[0][0]) / (pts_end[-1][1] - pts_end[0][1])
            else:
                prev_exit_slope = -999.0
        elif len(seg_pts) >= 2:
            if seg_pts[-1][1] != seg_pts[0][1]:
                prev_exit_slope = (seg_pts[-1][0] - seg_pts[0][0]) / (seg_pts[-1][1] - seg_pts[0][1])
            else:
                prev_exit_slope = -999.0
        else:
            prev_exit_slope = -999.0
            
        if all_points:
            seg_pts_to_add = seg_pts[1:]  # avoid duplicating the shared anchor row
        else:
            seg_pts_to_add = seg_pts
        all_points.extend(seg_pts_to_add)
        
        seg_conf = float(np.clip(1.0 - seg_cost, 0.0, 1.0))
        needs_anchor = seg_conf < 0.75
        
        segments.append({
            "from": list(p_start),
            "to": list(p_end),
            "mean_cost": round(seg_cost, 4),
            "confidence": round(seg_conf, 4),
            "worst_pt": worst_pt,
            "needs_anchor": needs_anchor,
        })
            
    return all_points, segments

# Define patched_gap_fill with original outlier filter
def patched_gap_fill(cost, p_start, p_end, x_min=-1, x_max=-1, prior_x=None, exclusive=False, est_gap_px=15.0, is_dotted=False, corridor_half_width=60, bridge_half_width=4):
    h, w = cost.shape
    ys, xe = p_start[1], p_end[0]
    ye, xs = p_end[1], p_start[0]
    if ys > ye:
        ys, ye = ye, ys
        xs, xe = xe, xs
        
    y_len = ye - ys + 1
    centroids = []
    
    # 1. Gather centroids
    for y in range(ys, ye + 1):
        if prior_x is not None:
            expected_x = prior_x[y - ys]
        else:
            frac = (y - ys) / (y_len - 1) if y_len > 1 else 0.0
            expected_x = xs + frac * (xe - xs)
            
        x_start = max(0, int(expected_x - corridor_half_width))
        x_end = min(w, int(expected_x + corridor_half_width + 1))
        
        row_segment = cost[y, x_start:x_end]
        ink_indices = np.where(row_segment < 0.45)[0]
        
        if len(ink_indices) > 0:
            centroid_x = x_start + np.mean(ink_indices)
            centroids.append((y, centroid_x))
            
    # 2. Reject outliers using local-median ± 3xMAD filter
    surviving = []
    if len(centroids) >= 5:
        ys_c = np.array([c[0] for c in centroids])
        xs_c = np.array([c[1] for c in centroids])
        
        window = 15
        half = window // 2
        keep = np.ones(len(centroids), dtype=bool)
        
        for i in range(len(centroids)):
            start = max(0, i - half)
            end = min(len(centroids), i + half + 1)
            local_xs = xs_c[start:end]
            med = np.median(local_xs)
            mad = np.median(np.abs(local_xs - med))
            # Relaxed outlier rejection to allow tracking sharp curves
            if is_dotted:
                thresh = max(6.0 * mad, 25.0)
            else:
                thresh = max(4.0 * mad, 15.0)
            if abs(xs_c[i] - med) > thresh:
                keep[i] = False
                
        surviving = [centroids[i] for i in range(len(centroids)) if keep[i]]
    else:
        surviving = centroids

    anchor_dict = {ys: xs, ye: xe}
    pts_dict = {int(round(y)): float(x) for y, x in surviving}
    pts_dict.update(anchor_dict)
    
    unique_ys = sorted(pts_dict.keys())
    unique_xs = [pts_dict[y] for y in unique_ys]
    
    trail_ys = np.arange(ys, ye + 1)
    if len(unique_ys) >= 4:
        try:
            from scipy.interpolate import PchipInterpolator
            pchip = PchipInterpolator(unique_ys, unique_xs)
            trail_xs = pchip(trail_ys)
        except Exception:
            trail_xs = np.interp(trail_ys, unique_ys, unique_xs)
    else:
        trail_xs = np.interp(trail_ys, unique_ys, unique_xs)
        
    if len(trail_xs) > 5:
        trail_xs = np.convolve(trail_xs, np.ones(5)/5, mode='same')
        trail_xs[0] = xs
        trail_xs[-1] = xe
        if len(trail_xs) > 2:
            trail_xs[1] = (trail_xs[0] + trail_xs[2]) / 2.0
            trail_xs[-2] = (trail_xs[-1] + trail_xs[-3]) / 2.0

    left_clip = max(0, x_min) if x_min >= 0 else 0
    right_clip = min(w - 1, x_max) if x_max >= 0 else (w - 1)
    trail_xs = np.clip(trail_xs, left_clip, right_clip)
    
    res_cost = cost.copy()
    R = float(bridge_half_width)
    gap_cost = 0.35 if is_dotted else 0.32
    x_coords = np.arange(0, cost.shape[1], dtype=np.float32)

    for idx, y in enumerate(trail_ys):
        x_mid = float(trail_xs[idx])
        x_start = max(0, int(x_mid - R))
        x_end = min(cost.shape[1], int(x_mid + R) + 1)
        d = np.abs(x_coords[x_start:x_end] - x_mid)
        in_valley = d <= R
        row = res_cost[y, x_start:x_end]
        
        # New formula
        valley_c = gap_cost + (1.0 - gap_cost) * (d / R)
        res_cost[y, x_start:x_end] = np.where(in_valley, np.minimum(row, valley_c), row)
        
        if exclusive:
            out_row = res_cost[y, :x_start]
            res_cost[y, :x_start] = np.where(out_row < 0.6, np.maximum(out_row, 0.75), out_row)
            out_row2 = res_cost[y, x_end:]
            res_cost[y, x_end:] = np.where(out_row2 < 0.6, np.maximum(out_row2, 0.75), out_row2)

    return res_cost

guided_curve_tracker._apply_anchor_guided_gap_fill = patched_gap_fill
guided_curve_tracker._trace_all_segments = patched_trace_all_segments

# Now run the tracking
res = guided_curve_tracker.trace_guided_curve(
    img,
    anchors,
    curve_style="dashed",
    corridor_pad=140,
    smooth_window=1,
)

traced_pts = res["points"]
traced_dict = {y: x for x, y in traced_pts}

# Print trace values and check errors
max_err = 0.0
for y in range(H):
    if y in traced_dict:
        err = abs(traced_dict[y] - xb[y])
        if err > max_err:
            max_err = err
        if err > 20:
            print(f"y={y}: traced_x={traced_dict[y]}, true_dashed={xb[y]}, error={err}")

print("Max error in fully patched run:", max_err)
