"""
Human-Guided Curve Tracker
==========================

The user clicks N anchor points on ONE curve in the well-log image.
This module then traces the *actual curve pixels* between every pair of
consecutive anchors using:

  1. Appearance learning  - the curve's color/darkness is sampled around the
                            user's anchor points (so it works for black curves,
                            red curves, dashed curves, ... whatever the user
                            clicked on).
  2. Cost map             - every pixel gets a cost = how unlike the target
                            curve it looks. Gridlines that don't match the
                            learned appearance get an extra penalty.
  3. Anchored optimal path- between each consecutive anchor pair a dynamic
                            programming shortest path is solved that is
                            guaranteed to (a) pass exactly through both
                            anchors and (b) produce exactly one X per depth
                            row (well-log curves are functions of depth).
                            This is globally optimal inside the corridor -
                            not a greedy tracker, so it cannot "wander off"
                            and it crosses other curves / gridlines correctly.

Exposed as a FastAPI router:  POST /guided-curve-track
"""

from __future__ import annotations

import io
import json
import hashlib
import numpy as np
from collections import OrderedDict
from typing import List, Optional, Tuple

# --- Guide Weight for Dashed Curves ---
DASHED_GUIDE_WEIGHT = 0.0002 # FIX 2: Expose the value as a module-level constant so it can be tuned
DASHED_CORRIDOR_HALF_WIDTH = 60 # FIX 3: Expose corridor half-width as a module-level constant
DASHED_BRIDGE_HALF_WIDTH = 4 # FIX 3: Expose bridge half-width as a module-level constant

# --- Image Cache for Faster Repeated Tracking ---
_IMAGE_CACHE: "OrderedDict[str, np.ndarray]" = OrderedDict()
_IMAGE_CACHE_MAX = 8

def _cache_image(bgr: np.ndarray) -> str:
    image_id = hashlib.sha1(bgr.tobytes()).hexdigest()[:16]
    if image_id in _IMAGE_CACHE:
        _IMAGE_CACHE.move_to_end(image_id)
    else:
        _IMAGE_CACHE[image_id] = bgr
        if len(_IMAGE_CACHE) > _IMAGE_CACHE_MAX:
            _IMAGE_CACHE.popitem(last=False)
    return image_id

def _get_cached_image(image_id: str) -> Optional[np.ndarray]:
    bgr = _IMAGE_CACHE.get(image_id)
    if bgr is not None:
        _IMAGE_CACHE.move_to_end(image_id)
    return bgr

import cv2

try:
    from fastapi import APIRouter, File, Form, HTTPException, UploadFile
    _HAS_FASTAPI = True
    router = APIRouter()
except ImportError:  # algorithm remains usable without the web layer
    _HAS_FASTAPI = False
    router = None

# --------------------------------------------------------------------------- #
#  Appearance model
# --------------------------------------------------------------------------- #

def _sample_curve_appearance(
    lab: np.ndarray,
    anchors: List[Tuple[int, int]],
    sample_radius: int = 9, # FIX 1: Use smaller default sample_radius (9, not 25)
) -> np.ndarray:
    """
    Learn what the target curve looks like (median LAB color) from small
    windows around the user's anchor points.

    User clicks are usually a few pixels off a 1-3 px wide curve, so the
    window is mostly paper. We therefore keep only pixels that genuinely
    stand out from the window's own background: inkness must exceed the
    window median by a clear margin.
    """
    h, w = lab.shape[:2]
    reps = [] # representative LAB color per anchor
    for (x, y) in anchors:
        x0, x1 = max(0, x - sample_radius), min(w, x + sample_radius + 1)
        y0, y1 = max(0, y - sample_radius), min(h, y + sample_radius + 1)
        patch = lab[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
        if patch.size == 0:
            continue
        # "ink-ness": dark (low L) or saturated (a/b far from neutral 128)
        L = patch[:, 0]
        chroma = np.abs(patch[:, 1] - 128) + np.abs(patch[:, 2] - 128)
        inkness = (255.0 - L) + 1.5 * chroma
        bg = np.median(inkness)               # window background level
        margin = max(30.0, 0.35 * (inkness.max() - bg))
        mask = inkness > bg + margin
        if not mask.any():
            continue
        # Get absolute distances to the click point for each pixel in patch
        yy, xx = np.mgrid[y0:y1, x0:x1]
        patch_spatial_dist = np.sqrt((xx - x) ** 2 + (yy - y) ** 2).reshape(-1)
        
        mask_indices = np.where(mask)[0]
        masked_dists = patch_spatial_dist[mask_indices]
        masked_inkness = inkness[mask_indices]
        
        # Sort by distance first, then by -inkness for tie-break toward higher inkness
        sort_keys = [(masked_dists[i], -masked_inkness[i], mask_indices[i]) for i in range(len(mask_indices))]
        sort_keys.sort()
        best_idx = sort_keys[0][2]
        
        reps.append(patch[best_idx]) # FIX 1: Pick the ONE representative pixel closest to click

    if not reps:
        # fallback: pure black ink model
        return np.array([0.0, 128.0, 128.0], dtype=np.float32)
        
    reps = np.array(reps, dtype=np.float32)
    # Reject outliers with a median + MAD test if there are at least 3 representatives
    if len(reps) >= 3:
        med_rep = np.median(reps, axis=0)
        l1_dists = np.sum(np.abs(reps - med_rep), axis=1)
        med_dist = np.median(l1_dists)
        mad = np.median(np.abs(l1_dists - med_dist))
        # drop reps whose L1 distance from the median exceeds 2.5 * MAD (with a small epsilon check)
        keep = l1_dists <= (2.5 * mad + 1e-5) # FIX 1: Reject outliers using median + MAD test
        surviving = reps[keep]
        if len(surviving) > 0:
            return np.median(surviving, axis=0).astype(np.float32) # FIX 1: Return the median of surviving reps
            
    return np.median(reps, axis=0).astype(np.float32) # FIX 1: Return median of all reps if too few


def _estimate_curve_style(bgr: np.ndarray, curve_lab: np.ndarray, anchors: List[Tuple[int, int]]) -> Tuple[str, float]:
    """
    Returns (detected_style, est_gap_px)
    """
    if len(anchors) < 2:
        return "solid", 0.0
    
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
    
    all_gaps = []
    all_dashes = []
    
    for i in range(len(anchors) - 1):
        x0, y0 = anchors[i]
        x1, y1 = anchors[i+1]
        
        length = int(np.hypot(x1 - x0, y1 - y0))
        if length == 0: continue
        
        num_pts = length * 2
        xs = np.linspace(x0, x1, num_pts)
        ys = np.linspace(y0, y1, num_pts)
        
        vals = []
        for x, y in zip(xs, ys):
            xi, yi = int(round(x)), int(round(y))
            y_min = max(0, yi-2)
            y_max = min(ink_mask.shape[0], yi+3)
            x_min = max(0, xi-2)
            x_max = min(ink_mask.shape[1], xi+3)
            if y_min < y_max and x_min < x_max:
                vals.append(np.max(ink_mask[y_min:y_max, x_min:x_max]))
            else:
                vals.append(0)
                
        vals = np.array(vals)
        padded = np.pad(vals, (1, 1), constant_values=0)
        diffs = np.diff(padded)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        
        runs = ends - starts
        all_dashes.extend(runs / 2.0)
        
        if len(ends) > 1:
            gaps = starts[1:] - ends[:-1]
            all_gaps.extend(gaps / 2.0)

    med_dash = float(np.median(all_dashes)) if len(all_dashes) > 0 else 0.0
    med_gap = float(np.median(all_gaps)) if len(all_gaps) > 0 else 0.0
    
    if med_gap <= 3.5:
        return "solid", 0.0
    elif med_dash <= 6.0 and med_gap <= 8.5:
        return "dotted", med_gap
    else:
        return "dashed", med_gap


def _build_cost_map(
    bgr: np.ndarray,
    curve_lab: np.ndarray,
    is_dashed: bool,
    suppress_gridlines: bool = True,
    est_gap_px: float = 15.0,
    is_dotted: bool = False,
) -> np.ndarray:
    """
    Build a pixel-wise cost map for pathfinding.
    Cost is high for background/grid and low for pixels matching curve_lab.
    Adaptive Physics: Gap bridging and solid-line erasure is ONLY applied if is_dashed is true.
    For dotted curves, blob penalisation uses a tighter threshold (8 px) so only
    marks larger than a dot (i.e. dashes or solid lines) get suppressed.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    # Increase tolerance for the L channel to handle faded ink
    target_chroma = np.sqrt(curve_lab[1]**2 + curve_lab[2]**2)
    if target_chroma < 12:
        wL, wa, wb = 2.5, 1.0, 1.0
    else:
        wL, wa, wb = 0.3, 2.5, 2.5

    dL = lab[:, :, 0] - curve_lab[0]
    # Penalize pixels lighter than the target heavily, but forgive pixels that are slightly darker ink
    dL = np.where(dL < 0, dL * 0.8, dL)
    
    da = lab[:, :, 1] - curve_lab[1]
    db = lab[:, :, 2] - curve_lab[2]
    
    dist = np.sqrt(wL * dL * dL + wa * da * da + wb * db * db)

    # Stricter normalization for enhanced color isolation
    cost = np.clip(dist / 60.0, 0.0, 1.0)

    # 1. ADAPTIVE SOLID LINE ERASURE - Pre-detection before grid suppression
    solid_mask = None
    non_dot_mask = None
    dot_mask = None

    if is_dashed:
        if is_dotted:
            # Dotted curves: isolate tiny dots (<= 8px)
            # Reconnect small gaps (e.g. anti-aliasing) but don't bridge dots
            close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
            closed_ink = cv2.morphologyEx((cost < 0.35).astype(np.uint8), cv2.MORPH_CLOSE, close_k)
            blob_thresh = 8  # 8px blob limit for dots
            _, core_ink = cv2.threshold(closed_ink.astype(np.float32) * 255, 127, 255, cv2.THRESH_BINARY)
            core_ink = core_ink.astype(np.uint8)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(core_ink, connectivity=8)
            
            non_dot_mask = np.zeros_like(cost, dtype=bool)
            dot_mask = np.zeros_like(cost, dtype=bool)
            for i in range(1, num_labels):
                h_i = stats[i, cv2.CC_STAT_HEIGHT]
                w_i = stats[i, cv2.CC_STAT_WIDTH]
                if h_i > blob_thresh or w_i > blob_thresh:
                    non_dot_mask[labels == i] = True
                else:
                    dot_mask[labels == i] = True
        else:
            # Dashed curves: penalize solid lines, but leave dashes alone
            close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
            closed_ink = cv2.morphologyEx((cost < 0.35).astype(np.uint8), cv2.MORPH_CLOSE, close_k)
            blob_thresh = 100  # Solid lines will be very long (e.g. > 100px)
            _, core_ink = cv2.threshold(closed_ink.astype(np.float32) * 255, 127, 255, cv2.THRESH_BINARY)
            core_ink = core_ink.astype(np.uint8)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(core_ink, connectivity=8)
            
            solid_mask = np.zeros_like(cost, dtype=bool)
            for i in range(1, num_labels):
                h_i = stats[i, cv2.CC_STAT_HEIGHT]
                w_i = stats[i, cv2.CC_STAT_WIDTH]
                if h_i > blob_thresh or w_i > blob_thresh:
                    solid_mask[labels == i] = True
        
    if suppress_gridlines:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape
        dark = (gray < 160).astype(np.uint8)
        
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        colored_ink = (sat > 40).astype(np.uint8)
        all_ink = np.bitwise_or(dark, colored_ink)

        # --- VERTICAL gridlines: columns that are dark ink across most of the height ---
        col_density = dark.sum(axis=0) / float(H)
        grid_cols = np.where(col_density > 0.55)[0]
        vgrid = np.zeros((H, W), dtype=bool)
        vgrid[:, grid_cols] = True

        # --- HORIZONTAL gridlines: rows that are ink across most of the width ---
        row_density = all_ink.sum(axis=1) / float(W)
        grid_rows = np.where(row_density > 0.55)[0]
        hgrid = np.zeros((H, W), dtype=bool)
        hgrid[grid_rows, :] = True

        # Safety net for broken/short gridlines: long thin straight runs only.
        vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(151, H // 6)))
        hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(151, W // 6), 1))
        vgrid |= cv2.morphologyEx(dark * 255, cv2.MORPH_OPEN, vk) > 0
        hgrid |= cv2.morphologyEx(all_ink * 255, cv2.MORPH_OPEN, hk) > 0

        grid = vgrid | hgrid

        # Unconditionally penalize grid lines so they are strictly avoided
        cost = np.where(grid, np.minimum(cost + 0.8, 1.0), cost)

    # 2. Apply pre-detected solid/dotted masks or solid curve smoothing
    if is_dashed:
        if is_dotted:
            if non_dot_mask is not None:
                cost = np.where(non_dot_mask, np.maximum(cost, 0.75), cost)
            if dot_mask is not None:
                cost = np.where(dot_mask & (cost > 0.0), 0.0, cost)
        else:
            if solid_mask is not None:
                cost = np.where(solid_mask, np.maximum(cost, 0.75), cost)
    else:
        # For solid curves, use a tiny 5-pixel bridge just to fix anti-aliasing artifacts.
        curve_mask = (cost < 0.45).astype(np.uint8)
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
        closed_mask = cv2.morphologyEx(curve_mask, cv2.MORPH_CLOSE, kernel_v)
        cost = np.where((closed_mask == 1) & (cost > 0.35), 0.35, cost)

    return cost.astype(np.float32)


# --------------------------------------------------------------------------- #
#  Anchored optimal path (dynamic programming, one X per row)
# --------------------------------------------------------------------------- #

def _snap_anchor(cost: np.ndarray, x: int, y: int, radius: int = 15) -> Tuple[int, int]:
    """Snap a user click to the lowest-cost pixel within the nearest connected ink component."""
    h, w = cost.shape
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    win = cost[y0:y1, x0:x1].copy()
    
    ink = (win < 0.45).astype(np.uint8)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(ink, connectivity=8)
    
    click_y, click_x = y - y0, x - x0
    target_label = labels[click_y, click_x] if click_y < win.shape[0] and click_x < win.shape[1] else 0
    
    if target_label == 0 and num > 1:
        min_dist = float('inf')
        for i in range(1, num):
            ys, xs = np.where(labels == i)
            if len(ys) > 0:
                dists = (ys - click_y)**2 + (xs - click_x)**2
                min_d = np.min(dists)
                if min_d < min_dist:
                    min_dist = min_d
                    target_label = i
                    
    if target_label > 0:
        win[labels != target_label] = 1.0

    yy, xx = np.mgrid[y0:y1, x0:x1]
    d2 = ((xx - x) ** 2 + (yy - y) ** 2).astype(np.float32)
    score = win + 0.0004 * d2
    idx = np.unravel_index(np.argmin(score), score.shape)
    return int(x0 + idx[1]), int(y0 + idx[0])



try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

if _HAS_NUMBA:
    @njit
    def _fast_dp_full(sub, dxs, move_pen, curvature_penalty, start_x, start_cost, endpoint_slope_weight, start_slope):
        rows, cols = sub.shape
        K = len(dxs)
        INF = np.float32(1e9)
        dp = np.full((cols, K), INF, dtype=np.float32)
        if start_slope == -999:
            dp[start_x, :] = start_cost
        else:
            for k in range(K):
                dp[start_x, k] = start_cost + endpoint_slope_weight * abs(dxs[k] - start_slope)
        parents = np.zeros((rows, cols, K), dtype=np.int16)
        A = np.zeros((cols, K), dtype=np.float32)
        arg = np.zeros((cols, K), dtype=np.int16)
        new_dp = np.zeros((cols, K), dtype=np.float32)
        for r in range(1, rows):
            for x in range(cols):
                for k in range(K):
                    A[x, k] = dp[x, k]
                    arg[x, k] = k
            for k in range(1, K):
                for x in range(cols):
                    if A[x, k - 1] + curvature_penalty < A[x, k]:
                        A[x, k] = A[x, k - 1] + curvature_penalty
                        arg[x, k] = arg[x, k - 1]
            for k in range(K - 2, -1, -1):
                for x in range(cols):
                    if A[x, k + 1] + curvature_penalty < A[x, k]:
                        A[x, k] = A[x, k + 1] + curvature_penalty
                        arg[x, k] = arg[x, k + 1]
            for x in range(cols):
                for k in range(K):
                    new_dp[x, k] = INF
            for k in range(K):
                dx = dxs[k]
                pen = move_pen[k]
                for x in range(cols):
                    src_x = x - dx
                    if 0 <= src_x < cols:
                        new_dp[x, k] = A[src_x, k] + pen
                        parents[r, x, k] = arg[src_x, k]
            for x in range(cols):
                cst = sub[r, x]
                for k in range(K):
                    dp[x, k] = new_dp[x, k] + cst
        return dp, parents



def _trace_segment_dp(
    cost: np.ndarray,
    p_start: Tuple[int, int],
    p_end: Tuple[int, int],
    max_slope_px: int,
    move_penalty: float,
    curvature_penalty: float,
    corridor_pad: int,
    start_slope: Optional[float] = None,
    end_slope: Optional[float] = None,
    endpoint_slope_weight: float = 0.0,
    x_min: int = -1,
    x_max: int = -1,
    guide_weight: float = 0.0, # FIX 2: Add guide_weight parameter with default 0.0
    trajectory: Optional[np.ndarray] = None, # FIX 2: Add trajectory parameter with default None
) -> Tuple[List[List[int]], float, List[int]]:
    """
    Globally-optimal path between two anchors with exactly one x per row,
    using a SECOND-ORDER model: the state is (x, incoming slope). Besides the
    pixel cost and a small |dx| movement penalty, slope CHANGES are penalized
    (curvature). This encodes "curve nature": at a crossing with another
    curve, the path prefers to continue in its current direction instead of
    switching branches.

    dp[r, x, k] = cost[r, x] + move_penalty*|dx_k|
                  + min_j ( dp[r-1, x - dx_k, j] + curvature_penalty*|dx_k - dx_j| )

    The min over j with an L1 slope-change penalty is computed in O(K) per x
    with a forward/backward distance-transform pass.

    Returns (points [[x, y], ...] inclusive of both anchors, mean path cost).
    """
    (xs, ys), (xe, ye) = p_start, p_end
    if ye < ys:
        (xs, ys), (xe, ye) = (xe, ye), (xs, ys)
    if ye == ys:  # degenerate: same row -> straight horizontal hop
        xs2, xe2 = sorted((xs, xe))
        pts = [[x, ys] for x in range(xs2, xe2 + 1)] or [[xs, ys]]
        c = float(np.mean([cost[ys, p[0]] for p in pts]))
        worst_pt = pts[np.argmax([cost[ys, p[0]] for p in pts])]
        return pts, c, worst_pt

    h, w = cost.shape
    cx0 = max(0, min(xs, xe) - corridor_pad)
    cx1 = min(w, max(xs, xe) + corridor_pad + 1)
    if x_min >= 0: cx0 = max(cx0, x_min)
    if x_max >= 0: cx1 = min(cx1, x_max + 1)
    sub = cost[ys:ye + 1, cx0:cx1]
    rows, cols = sub.shape
    
    if guide_weight > 0.0 and trajectory is not None:
        sub = sub.copy() # FIX 2: Copy to avoid modifying the original cost map
        x_target = trajectory - cx0
        col_indices = np.arange(cols)
        pull = guide_weight * (col_indices[None, :] - x_target[:, None])**2
        sub += pull.astype(np.float32) # FIX 2: Apply quadratic pull toward anchor trajectory

    S = max_slope_px
    K = 2 * S + 1
    dxs = np.arange(-S, S + 1, dtype=np.int32)           # slope per state k
    move_pen = (move_penalty * np.abs(dxs)).astype(np.float32)

    INF = np.float32(1e9)
    if _HAS_NUMBA:
        dp, parents = _fast_dp_full(
            sub, dxs, move_pen, curvature_penalty, 
            xs - cx0, sub[0, xs - cx0], endpoint_slope_weight, 
            start_slope if start_slope is not None else -999.0
        )
    else:
        dp = np.full((cols, K), INF, dtype=np.float32)
        if start_slope is None:
            dp[xs - cx0, :] = sub[0, xs - cx0]                   # start: any slope
        else:
            for k in range(K):
                dp[xs - cx0, k] = sub[0, xs - cx0] + endpoint_slope_weight * abs(dxs[k] - start_slope)
        parents = np.zeros((rows, cols, K), dtype=np.int16)

        for r in range(1, rows):
            A = dp.copy()
            arg = np.tile(np.arange(K, dtype=np.int16), (cols, 1))
            for k in range(1, K):                            # forward pass
                better = A[:, k - 1] + curvature_penalty < A[:, k]
                A[better, k] = A[better, k - 1] + curvature_penalty
                arg[better, k] = arg[better, k - 1]
            for k in range(K - 2, -1, -1):                   # backward pass
                better = A[:, k + 1] + curvature_penalty < A[:, k]
                A[better, k] = A[better, k + 1] + curvature_penalty
                arg[better, k] = arg[better, k + 1]

            new_dp = np.full((cols, K), INF, dtype=np.float32)
            par_r = parents[r]
            for k in range(K):
                dx = dxs[k]
                if dx >= 0:
                    src = slice(0, cols - dx) if dx > 0 else slice(0, cols)
                    dst = slice(dx, cols) if dx > 0 else slice(0, cols)
                else:
                    src = slice(-dx, cols)
                    dst = slice(0, cols + dx)
                new_dp[dst, k] = A[src, k] + move_pen[k]
                par_r[dst, k] = arg[src, k]
            new_dp += sub[r][:, None]
            dp = new_dp

    xe_l = xe - cx0
    k = int(np.argmin(dp[xe_l]))
    total = dp[xe_l, k]
    if not np.isfinite(total):
        # corridor too narrow / slope too steep -> linear fallback
        n = ye - ys + 1
        xs_lin = np.round(np.linspace(xs, xe, n)).astype(int)
        pts = [[int(xs_lin[i]), ys + i] for i in range(n)]
        
        # Calculate real cost of this fallback line
        costs = [cost[p[1], p[0]] for p in pts]
        c = float(np.mean(costs))
        worst_pt = pts[np.argmax(costs)]
        return pts, c, worst_pt

    # Backtrack: at (row r, x, slope k) we arrived via dx_k from x - dx_k,
    # with previous slope parents[r, x, k].
    x = xe_l
    path_local = [(x, rows - 1)]
    for r in range(rows - 1, 0, -1):
        pk = int(parents[r, x, k])
        x = int(np.clip(x - dxs[k], 0, cols - 1))
        k = pk
        path_local.append((x, r - 1))
    path_local.reverse()

    h, w = cost.shape
    pts = [[min(max(x + cx0, 0), w - 1), min(max(ys + r, 0), h - 1)] for x, r in path_local]
    costs = [cost[p[1], p[0]] for p in pts]
    c = float(np.mean(costs))
    worst_pt = pts[np.argmax(costs)]
    
    return pts, c, worst_pt


def _smooth_x(points: List[List[int]], window: int = 5) -> List[List[int]]:
    # Filter out None sentinel values (wrap-jump markers) before smoothing
    real_points = [p for p in points if p is not None]
    if len(real_points) < window:
        return points
    xs = np.array([p[0] for p in real_points], dtype=np.float32)
    from scipy.signal import medfilt
    w = window if window % 2 == 1 else window + 1
    xs_s = medfilt(xs, kernel_size=w)
    # Rebuild result, re-inserting None sentinels at original positions
    smoothed_iter = iter([[int(round(xs_s[i])), real_points[i][1]] for i in range(len(real_points))])
    return [next(smoothed_iter) if p is not None else None for p in points]


# --------------------------------------------------------------------------- #
#  Public tracing entry point
# --------------------------------------------------------------------------- #

def _apply_anchor_guided_gap_fill(
    cost: np.ndarray,
    p_start: Tuple[int, int],
    p_end: Tuple[int, int],
    x_min: int = -1,
    x_max: int = -1,
    prior_x: Optional[np.ndarray] = None,
    exclusive: bool = False,
    est_gap_px: float = 15.0,
    is_dotted: bool = False,
    corridor_half_width: int = 60, # FIX 3: Added corridor_half_width parameter with default 60
    bridge_half_width: int = 4, # FIX 3: Added bridge_half_width parameter with default 4
) -> np.ndarray:
    """
    Perform anchor-guided gap filling for dashed curves.
    Identifies centroids of dash segments, filters outliers, fits a PCHIP trail,
    and applies a graded cost valley along the trail to guide pathfinding.
    """
    h, w = cost.shape
    (xs, ys), (xe, ye) = p_start, p_end
    if ye < ys:
        (xs, ys), (xe, ye) = (xe, ye), (xs, ys)
    
    y_len = ye - ys + 1
    if y_len <= 1:
        return cost.copy()

    # 1. Collect per-row dash centroids
    corridor_pad = corridor_half_width # FIX 3: Use the provided corridor_half_width parameter
    centroids = []
    
    for y in range(ys, ye + 1):
        if prior_x is not None:
            expected_x = prior_x[y - ys]
        else:
            frac = (y - ys) / (y_len - 1) if y_len > 1 else 0.0
            expected_x = xs + frac * (xe - xs)
            
        x_start = max(0, int(expected_x - corridor_pad))
        x_end = min(w, int(expected_x + corridor_pad + 1))
        
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

    # Ensure start and end anchors are present and override any nearby centroids
    anchor_dict = {ys: xs, ye: xe}
    pts_dict = {int(round(y)): float(x) for y, x in surviving}
    pts_dict.update(anchor_dict)
    
    unique_ys = sorted(pts_dict.keys())
    unique_xs = [pts_dict[y] for y in unique_ys]
    
    # 3. Fit PCHIP trail
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
        
    # Lightly smooth the trail to reduce jitter
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
    
    # 4. Modify the cost map using vectorised NumPy operations
    filled_cost = cost.copy()
    
    # Adaptive gap parameters based on style
    R = float(bridge_half_width) # FIX 3: Use the provided bridge_half_width parameter
    gap_cost = 0.35 if is_dotted else 0.32
    
    x_coords = np.arange(0, cost.shape[1], dtype=np.float32)

    for idx, y in enumerate(trail_ys):
        x_mid = float(trail_xs[idx])
        x_start = max(0, int(x_mid - R))
        x_end = min(cost.shape[1], int(x_mid + R) + 1)
        d = np.abs(x_coords[x_start:x_end] - x_mid)
        in_valley = d <= R
        row = filled_cost[y, x_start:x_end]
        valley_c = gap_cost + (1.0 - gap_cost) * (d / R)
        filled_cost[y, x_start:x_end] = np.where(in_valley, np.minimum(row, valley_c), row)
        if exclusive:
            # penalise pixels outside the valley
            out_row = filled_cost[y, :x_start]
            filled_cost[y, :x_start] = np.where(out_row < 0.6, np.maximum(out_row, 0.75), out_row)
            out_row2 = filled_cost[y, x_end:]
            filled_cost[y, x_end:] = np.where(out_row2 < 0.6, np.maximum(out_row2, 0.75), out_row2)

    return filled_cost


def _trace_all_segments(
    base_cost: np.ndarray,
    snapped: List[Tuple[int, int]],
    is_dashed: bool,
    max_slope_px: int,
    move_penalty: float,
    curvature_penalty: float,
    corridor_pad: int,
    x_min: int,
    x_max: int,
    prior_trajectory: Optional[List[List[int]]] = None,
    exclusive: bool = False,
    est_gap_px: float = 15.0,
    is_dotted: bool = False,
) -> Tuple[List[List[int]], List[dict]]:
    all_points: List[List[int]] = []
    segments = []
    prev_exit_slope = -999.0
    
    prior_map = None
    if prior_trajectory is not None:
        prior_map = {pt[1]: pt[0] for pt in prior_trajectory if pt is not None}
    # ---------------------------------------------------------------------------
    #  New preprocessing for dashed/dotted curves
    # ---------------------------------------------------------------------------
    # Configurable thresholds (can be exposed later via UI)
    SEGMENT_VERTICAL_TOL = 15          # max vertical distance between neighboring segments (pixels)
    SEGMENT_HORIZONTAL_TOL = 30        # max horizontal offset for continuity
    SEGMENT_ORIENT_TOL = 15            # max orientation difference (degrees)
    DEBUG_OUTPUT = False               # set True only when debugging dash detection
    DEBUG_DIR = "debug_output"

    import os
    if DEBUG_OUTPUT and not os.path.isdir(DEBUG_DIR):
        os.makedirs(DEBUG_DIR, exist_ok=True)

    def _detect_dash_dot_segments(cost_map: np.ndarray, corridor_mask: np.ndarray) -> List[dict]:
        """Detect dash/dot segments inside the corridor.
        Returns a list of dicts with keys: centroid, bbox, area, orientation, mask indices.
        """
        # Binary mask of ink within corridor
        ink_mask = (cost_map < 0.45).astype(np.uint8) * corridor_mask.astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ink_mask, connectivity=8)
        segments = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 3:  # ignore tiny noise
                continue
            x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                           stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
            mask_indices = np.where(labels == i)
            # orientation via moments
            moments = cv2.moments((labels == i).astype(np.uint8))
            if moments["mu20"] != 0:
                orientation = 0.5 * np.arctan2(2 * moments["mu11"], moments["mu20"] - moments["mu02"]) * 180 / np.pi
            else:
                orientation = 0.0
            segments.append({
                "centroid": (centroids[i][0], centroids[i][1]),
                "bbox": (x, y, w, h),
                "area": area,
                "orientation": orientation,
                "mask_idx": mask_indices,
                "label": i,
            })
        if DEBUG_OUTPUT:
            # draw each segment in a random colour for debugging
            dbg = np.zeros_like(cost_map, dtype=np.uint8)
            for seg in segments:
                color = np.random.randint(50, 255, size=3, dtype=np.uint8)
                lbl = seg["label"]
                dbg[labels == lbl] = color[0]
            cv2.imwrite(os.path.join(DEBUG_DIR, "detected_segments.png"), dbg)
        return segments

    def _group_segments(segments: List[dict]) -> List[List[dict]]:
        """Group neighboring segments into curves based on geometry heuristics."""
        if not segments:
            return []
        # sort by centroid y (depth)
        segments = sorted(segments, key=lambda s: s["centroid"][1])
        groups = []
        current = [segments[0]]
        for seg in segments[1:]:
            prev = current[-1]
            dy = abs(seg["centroid"][1] - prev["centroid"][1])
            dx = abs(seg["centroid"][0] - prev["centroid"][0])
            d_orient = abs(seg["orientation"] - prev["orientation"])
            if dy <= SEGMENT_VERTICAL_TOL and dx <= SEGMENT_HORIZONTAL_TOL and d_orient <= SEGMENT_ORIENT_TOL:
                current.append(seg)
            else:
                groups.append(current)
                current = [seg]
        groups.append(current)
        return groups

    def _build_graph_from_group(group: List[dict]) -> List[int]:
        """Simple graph: each node connects to the next feasible node (by depth). Returns an ordering of segment indices.
        For now we just order by centroid y.
        """
        ordered = sorted(range(len(group)), key=lambda i: group[i]["centroid"][1])
        return ordered

    def _reconstruct_curve_from_group(group: List[dict]) -> List[Tuple[int, int]]:
        """Create a virtual continuous curve from grouped segments using spline interpolation.
        Returns list of (x, y) integer points sampled at each integer row.
        """
        if not group:
            return []
        centroids = np.array([seg["centroid"] for seg in group], dtype=np.float32)
        order = np.argsort(centroids[:, 1])
        xs = centroids[order, 0]
        ys = centroids[order, 1]
        from scipy.interpolate import PchipInterpolator
        y_min = int(np.round(ys[0]))
        y_max = int(np.round(ys[-1]))
        trail_ys = np.arange(y_min, y_max + 1)
        try:
            pchip = PchipInterpolator(ys, xs)
            trail_xs = pchip(trail_ys)  # vectorised — no Python loop
        except Exception:
            trail_xs = np.interp(trail_ys, ys, xs)
        return [[int(round(float(trail_xs[i]))), int(trail_ys[i])] for i in range(len(trail_ys))]

    def _preprocess_dash_dotted(cost: np.ndarray, corridor_pad: int, is_dashed: bool, is_dotted: bool) -> List[Tuple[int, int]]:
        """Full preprocessing pipeline for dashed/dotted curves.
        Returns a list of (x, y) points representing a reconstructed continuous curve.
        """
        # create a mask for the corridor (used later for component analysis)
        h, w = cost.shape
        corridor_mask = np.zeros_like(cost, dtype=bool)
        # use a generous vertical window around the line (already defined by corridor_pad)
        # Here we simply mark the entire image – segmentation will later filter by area.
        corridor_mask[:, :] = True
        segments = _detect_dash_dot_segments(cost, corridor_mask)
        groups = _group_segments(segments)
        # stats collection
        total_segments = len(segments)
        rows_with_pixels = np.count_nonzero(cost < 0.45)
        reconstructed_pts = []
        cover_rows = set()
        gaps = []
        for grp in groups:
            ordered_idx = _build_graph_from_group(grp)
            ordered_segs = [grp[i] for i in ordered_idx]
            pts = _reconstruct_curve_from_group(ordered_segs)
            if pts:
                reconstructed_pts.extend(pts)
                # record rows and gaps for stats
                rows = [p[1] for p in pts]
                cover_rows.update(rows)
                if len(rows) > 1:
                    # compute gaps between consecutive rows
                    gaps.extend([rows[i+1] - rows[i] - 1 for i in range(len(rows)-1)])
        # Debug overlay of reconstructed curve
        if DEBUG_OUTPUT:
            overlay = np.dstack([cost * 255] * 3).astype(np.uint8)
            for pt in reconstructed_pts:
                cv2.circle(overlay, (pt[0], pt[1]), 2, (0, 255, 0), -1)
            cv2.imwrite(os.path.join(DEBUG_DIR, "reconstructed_curve.png"), overlay)
        # Compute statistics (could be returned via a global dict or printed)
        coverage_pct = (len(cover_rows) / h) * 100 if h > 0 else 0
        avg_gap = np.mean(gaps) if gaps else 0
        max_gap = np.max(gaps) if gaps else 0
        stats = {
            "total_segments": total_segments,
            "rows_with_curve_pixels": rows_with_pixels,
            "reconstructed_coverage_pct": coverage_pct,
            "average_gap_size": avg_gap,
            "maximum_gap_size": max_gap,
        }
        # Store stats globally for later use (simple approach)
        global _PREPROCESS_STATS
        _PREPROCESS_STATS = stats
        return reconstructed_pts

    for i in range(len(snapped) - 1):
        p_start = snapped[i]
        p_end = snapped[i + 1]
        (xs_s, ys_s) = p_start
        (xe_s, ye_s) = p_end
        dy = abs(ye_s - ys_s)
        dx = abs(xe_s - xs_s)
        
        track_w = x_max - x_min if x_max > x_min else 800
        # For dashed curves, we do not treat large horizontal jumps as wrap jumps.
        # Otherwise, a genuine large gap would be broken into separate fragments.
        if is_dashed:
            is_wrap_jump = False
        else:
            track_left = x_min if x_min >= 0 else 0
            track_right = x_max if x_max >= 0 else base_cost.shape[1] - 1
            edge_tol = track_w * 0.15
            is_edge_jump = (min(xs_s, xe_s) - track_left < edge_tol) and (track_right - max(xs_s, xe_s) < edge_tol)
            is_wrap_jump = (dx > track_w * 0.4) and (dx > dy * 2) and is_edge_jump

        if is_wrap_jump:
            # Massive horizontal jump - do not force a connection!
            seg_pts = [list(p_start), None, list(p_end)]
            seg_cost = 0.0  # cost 0 prevents "needs anchor" warning
            worst_pt = None
            prev_exit_slope = -999.0
        else:
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
                
                seg_cost_map = _apply_anchor_guided_gap_fill(
                    base_cost, p_start, p_end, x_min=x_min, x_max=x_max,
                    prior_x=seg_prior_x, exclusive=exclusive, est_gap_px=est_gap_px,
                    is_dotted=is_dotted,
                    corridor_half_width=corridor_pad,
                    bridge_half_width=DASHED_BRIDGE_HALF_WIDTH
                )
                seg_guide_weight = DASHED_GUIDE_WEIGHT if prior_map is not None else 0.0
                seg_trajectory = seg_prior_x if prior_map is not None else None
            else:
                seg_cost_map = base_cost
                seg_guide_weight = 0.0
                seg_trajectory = None

            seg_pts, seg_cost, worst_pt = _trace_segment_dp(
                seg_cost_map, p_start, p_end,
                max_slope_px=seg_max_slope,
                move_penalty=move_penalty,
                curvature_penalty=seg_curvature_penalty,
                corridor_pad=int(min(corridor_pad, dx + 40)) if dy > 0 else corridor_pad,
                start_slope=prev_exit_slope if prev_exit_slope != -999.0 else None,
                endpoint_slope_weight=0.04 if prev_exit_slope != -999.0 else 0.0,
                x_min=x_min,
                x_max=x_max,
                guide_weight=seg_guide_weight, # FIX 2: Pass guide_weight
                trajectory=seg_trajectory, # FIX 2: Pass trajectory
            )
        
        if is_wrap_jump:
            pass # Keep prev_exit_slope = -999.0
        elif len(seg_pts) >= 8:
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
            seg_pts = seg_pts[1:]  # avoid duplicating the shared anchor row
        all_points.extend(seg_pts)
        
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


def trace_guided_curve(
    bgr: np.ndarray,
    anchors_xy: List[Tuple[int, int]],
    snap_radius: int = 15,
    max_slope_px: int = 35,
    move_penalty: float = 0.02,
    curvature_penalty: float = 0.06,
    corridor_pad: int = 120,
    smooth_window: int = 5,
    suppress_gridlines: bool = True,
    curve_style: str = "auto",
    x_min: int = -1,
    x_max: int = -1,
    occupied_points: Optional[List[Tuple[int, int]]] = None,
) -> dict:
    if len(anchors_xy) < 2:
        raise ValueError("At least 2 anchor points are required")

    h, w = bgr.shape[:2]
    left_clip = max(0, x_min) if x_min >= 0 else 0
    right_clip = min(w - 1, x_max) if x_max >= 0 else (w - 1)
    anchors = [(int(np.clip(x, left_clip, right_clip)), int(np.clip(y, 0, h - 1)))
               for x, y in anchors_xy]

    min_y = min(y for _, y in anchors)
    max_y = max(y for _, y in anchors)
    pad_y = 50
    crop_y_start = max(0, min_y - pad_y)
    crop_y_end = min(h, max_y + pad_y + 1)
    
    bgr_crop = bgr[crop_y_start:crop_y_end, :]
    anchors_crop = [(x, y - crop_y_start) for x, y in anchors]

    lab = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    curve_lab = _sample_curve_appearance(lab, anchors_crop)
    
    # 1. EXPLICIT STYLE OVERRIDE
    if curve_style in ("solid", "dashed", "dotted"):
        is_dotted = (curve_style == "dotted")
        is_dashed = (curve_style == "dashed") or is_dotted
        style_source = "user"
        est_gap_px = 15.0 if curve_style == "dashed" else (5.0 if is_dotted else 0.0)
        detected_style = curve_style
    else:                                               # "auto" (make this the new default)
        detected_style, est_gap_px = _estimate_curve_style(bgr_crop, curve_lab, anchors_crop)
        is_dashed = (detected_style in ("dashed", "dotted"))
        is_dotted = (detected_style == "dotted")
        style_source = "auto"
    
    # Build cost map with adaptive gap bridging and solid line erasure
    cost = _build_cost_map(bgr_crop, curve_lab, is_dashed, suppress_gridlines, est_gap_px, is_dotted=is_dotted)

    # 1.5. MUTUAL EXCLUSION: Penalize already-traced curves
    if occupied_points:
        for ox, oy in occupied_points:
            oy_crop = oy - crop_y_start
            if 0 <= oy_crop < cost.shape[0] and 0 <= ox < cost.shape[1]:
                y_min, y_max = max(0, oy_crop-1), min(cost.shape[0], oy_crop+2)
                x_min, x_max = max(0, ox-2), min(cost.shape[1], ox+3)
                cost[y_min:y_max, x_min:x_max] = np.maximum(cost[y_min:y_max, x_min:x_max], 0.95)

    # 2. ADAPTIVE PHYSICS
    if is_dashed:
        curvature_penalty = 0.15  # Extremely stiff to blast through erased solid lines
    else:
        curvature_penalty = 0.05  # Highly flexible to follow tight bends

    effective_snap_radius = min(snap_radius, 8) if is_dashed else snap_radius # FIX 3: Cap snap radius to ~8 in dashed mode to avoid snapping to wrong curves
    snapped_crop = [_snap_anchor(cost, x, y, effective_snap_radius) for (x, y) in anchors_crop]
    # Well-log curves are functions of depth: order anchors top -> bottom.
    snapped_crop = sorted(snapped_crop, key=lambda p: p[1])
    # Drop duplicates on the same row.
    dedup = [snapped_crop[0]]
    for p in snapped_crop[1:]:
        if p[1] != dedup[-1][1]:
            dedup.append(p)
    snapped_crop = dedup
    if len(snapped_crop) < 2:
        raise ValueError("Anchors collapsed to a single row - click points further apart vertically")

    # 3. SAME-COLOUR CURVE SEPARATION PRIOR
    if not is_dashed:
        ink = (cost < 0.45).astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        anchor_labels = {labels[y, x] for (x, y) in snapped_crop if labels[y, x] != 0}
        if anchor_labels:
            on_target = np.isin(labels, list(anchor_labels))
            cost = np.where(on_target, cost * 0.3, cost)

    # Run pass 1
    pts_1, segs_1 = _trace_all_segments(
        cost, snapped_crop, is_dashed, max_slope_px, move_penalty, curvature_penalty, corridor_pad, x_min, x_max,
        prior_trajectory=None, exclusive=False, est_gap_px=est_gap_px, is_dotted=is_dotted
    )

    if is_dashed:
        # Run pass 2
        pts_2, segs_2 = _trace_all_segments(
            cost, snapped_crop, is_dashed, max_slope_px, move_penalty, curvature_penalty, corridor_pad, x_min, x_max,
            prior_trajectory=pts_1, exclusive=True, est_gap_px=est_gap_px, is_dotted=is_dotted
        )

        # Apply the fair-cost guard segment-by-segment
        final_points = []
        final_segments = []
        
        for i in range(len(snapped_crop) - 1):
            p_start = snapped_crop[i]
            p_end = snapped_crop[i + 1]
            (xs_s, ys_s) = p_start
            (xe_s, ye_s) = p_end
            
            # Slice points belonging to this segment
            seg_pts_1 = [pt for pt in pts_1 if ys_s <= pt[1] <= ye_s]
            seg_pts_2 = [pt for pt in pts_2 if ys_s <= pt[1] <= ye_s]
            
            cost_map_1 = _apply_anchor_guided_gap_fill(
                cost, p_start, p_end, x_min=x_min, x_max=x_max,
                prior_x=None, exclusive=False, est_gap_px=est_gap_px,
                is_dotted=is_dotted,
                corridor_half_width=corridor_pad,
                bridge_half_width=DASHED_BRIDGE_HALF_WIDTH
            )
            
            eval_cost_1 = float(np.mean([cost_map_1[pt[1], pt[0]] for pt in seg_pts_1])) if seg_pts_1 else 1.0
            eval_cost_2 = float(np.mean([cost_map_1[pt[1], pt[0]] for pt in seg_pts_2])) if seg_pts_2 else 1.0
            
            if eval_cost_2 <= eval_cost_1:
                chosen_pts = seg_pts_2
                chosen_seg = segs_2[i]
            else:
                chosen_pts = seg_pts_1
                chosen_seg = segs_1[i]
                
            if final_points:
                chosen_pts = [pt for pt in chosen_pts if pt[1] > final_points[-1][1]]
            final_points.extend(chosen_pts)
            final_segments.append(chosen_seg)
            
        all_points = final_points
        segments = final_segments
    else:
        all_points = pts_1
        segments = segs_1

    # Strip None sentinels before smoothing (they mark wrap-jump breaks)
    all_points_clean = [p for p in all_points if p is not None]
    all_points_clean = _smooth_x(all_points_clean, smooth_window)
    # Rebuild with Nones reinserted
    none_positions = {i for i, p in enumerate(all_points) if p is None}
    out_iter = iter(all_points_clean)
    all_points = [None if i in none_positions else next(out_iter) for i in range(len(all_points))]
    
    for pt in all_points:
        if pt is not None:
            pt[1] += crop_y_start
            
    for s in segments:
        s["from"][1] += crop_y_start
        s["to"][1] += crop_y_start
        if s["worst_pt"] is not None:
            s["worst_pt"][1] += crop_y_start

    snapped = [[p[0], p[1] + crop_y_start] for p in snapped_crop]
    
    overall_conf = float(np.mean([s["confidence"] for s in segments]))

    # Edge logic for wrap detection
    top_pt = all_points[0]
    bottom_pt = all_points[-1]
    edge_in = None
    edge_out = None
    
    left_bound = x_min if x_min >= 0 else 0
    right_bound = x_max if x_max >= 0 else (w - 1)
    edge_tol = round(0.06 * (x_max - x_min)) if x_min >= 0 and x_max >= 0 else 10
    
    if abs(top_pt[0] - left_bound) <= edge_tol:
        edge_in = 'left'
    elif abs(top_pt[0] - right_bound) <= edge_tol:
        edge_in = 'right'
        
    if abs(bottom_pt[0] - left_bound) <= edge_tol:
        edge_out = 'left'
    elif abs(bottom_pt[0] - right_bound) <= edge_tol:
        edge_out = 'right'

    return {
        "points": all_points,                       # [[x, y], ...] one per row
        "snapped_anchors": [list(p) for p in snapped],
        "segments": segments,                       # per-segment confidence
        "confidence": round(overall_conf, 4),
        "curve_color_lab": [round(float(v), 1) for v in curve_lab],
        "num_points": len(all_points),
        "detected_style": detected_style,
        "style_source": style_source,
        "edge_in": edge_in,
        "edge_out": edge_out,
    }


def detect_wrap_connectors(points: List[List[int]], x_min: int, x_max: int, max_slope_px: int = 8):
    connectors = []
    if len(points) < 2 or x_min < 0 or x_max < 0:
        return connectors
        
    track_width = x_max - x_min
    if track_width <= 0:
        return connectors
        
    n = len(points)
    i = 0
    while i < n - 1:
        start_x, start_y = points[i]
        j = i + 1
        while j < n:
            dy = points[j][1] - points[j-1][1]
            dx = points[j][0] - points[j-1][0]
            slope = abs(dx / dy) if dy != 0 else float('inf')
            if slope < max_slope_px:
                break
            j += 1
            
        if j > i + 1:
            end_x, end_y = points[j-1]
            dx_total = end_x - start_x
            if abs(dx_total) >= 0.6 * track_width:
                edge_tol = 0.1 * track_width
                if start_x < end_x: # L->R
                    if (start_x - x_min) <= edge_tol and (x_max - end_x) <= edge_tol:
                        connectors.append((start_y, end_y, 'L->R'))
                else: # R->L
                    if (x_max - start_x) <= edge_tol and (end_x - x_min) <= edge_tol:
                        connectors.append((start_y, end_y, 'R->L'))
            i = j
        else:
            i += 1
    return connectors


# --------------------------------------------------------------------------- #
#  FastAPI endpoint
# --------------------------------------------------------------------------- #

def _register_endpoint():
    @router.post("/guided-curve-track")
    async def guided_curve_track(
        file: Optional[UploadFile] = File(None),
        image_id: Optional[str] = Form(None),
        points: str = Form(...),            # JSON: [[x, y], [x, y], ...] image-pixel coords
        snap_radius: int = Form(15),
        max_slope_px: int = Form(35),
        move_penalty: float = Form(0.02),
        curvature_penalty: float = Form(0.06),
        corridor_pad: int = Form(120),
        smooth_window: int = Form(5),
        suppress_gridlines: bool = Form(True),
        curve_style: str = Form("solid"),
        x_min: int = Form(-1),
        x_max: int = Form(-1),
        occupied_points: Optional[str] = Form(None),
    ):
        """
        Human-guided AI curve tracing.

        The frontend sends the displayed image plus the user's anchor clicks
        (in image pixel coordinates). Returns one traced point per depth row,
        passing exactly through every (snapped) anchor.
        """
        if image_id:
            bgr = _get_cached_image(image_id)
            if bgr is None:
                raise HTTPException(409, "image_id expired - re-upload the image")
        else:
            if file is None:
                raise HTTPException(400, "Provide either file or image_id")
            ext = (file.filename or "img.png").lower().rsplit(".", 1)[-1]
            if ext not in ("tif", "tiff", "png", "jpg", "jpeg", "bmp", "webp"):
                raise HTTPException(400, "Unsupported image format")
    
            data = await file.read()
            arr = np.frombuffer(data, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                raise HTTPException(400, "Failed to decode image")
            image_id = _cache_image(bgr)

        try:
            anchors = json.loads(points)
            anchors = [(int(round(p[0])), int(round(p[1]))) for p in anchors]
        except Exception:
            raise HTTPException(400, "points must be JSON like [[x,y],[x,y],...]")

        if len(anchors) < 2:
            raise HTTPException(400, "At least 2 anchor points are required")

        occ_pts = None
        if occupied_points:
            try:
                occ = json.loads(occupied_points)
                occ_pts = [(int(round(p[0])), int(round(p[1]))) for p in occ]
            except Exception:
                pass

        try:
            result = trace_guided_curve(
                bgr, anchors,
                snap_radius=snap_radius,
                max_slope_px=max_slope_px,
                move_penalty=move_penalty,
                curvature_penalty=curvature_penalty,
                corridor_pad=corridor_pad,
                smooth_window=smooth_window,
                suppress_gridlines=suppress_gridlines,
                curve_style=curve_style,
                x_min=x_min,
                x_max=x_max,
                occupied_points=occ_pts,
            )
        except ValueError as e:
            raise HTTPException(400, str(e))

        result["image_id"] = image_id
        return result

    @router.post("/detect-wraps")
    async def detect_wraps_endpoint(
        points: str = Form(...),            # JSON: [[x, y], [x, y], ...]
        x_min: int = Form(...),
        x_max: int = Form(...),
    ):
        try:
            pts = json.loads(points)
            pts = [[int(round(p[0])), int(round(p[1]))] for p in pts]
        except Exception:
            raise HTTPException(400, "points must be JSON like [[x,y],[x,y],...]")
            
        connectors = detect_wrap_connectors(pts, x_min, x_max)
        
        # Split points into fragments
        fragments = []
        wraps = []
        
        last_idx = 0
        current_wrap_level = 0
        edge_tol = round(0.06 * (x_max - x_min)) if x_min >= 0 and x_max >= 0 else 10
        
        for conn_start_y, conn_end_y, direction in connectors:
            # find index in pts matching conn_start_y
            frag_pts = []
            while last_idx < len(pts) and pts[last_idx][1] < conn_start_y:
                frag_pts.append(pts[last_idx])
                last_idx += 1
                
            if frag_pts:
                top_pt = frag_pts[0]
                bottom_pt = frag_pts[-1]
                
                edge_in = 'left' if abs(top_pt[0] - x_min) <= edge_tol else ('right' if abs(top_pt[0] - x_max) <= edge_tol else None)
                edge_out = 'left' if abs(bottom_pt[0] - x_min) <= edge_tol else ('right' if abs(bottom_pt[0] - x_max) <= edge_tol else None)
                
                fragments.append({
                    "points": frag_pts,
                    "wrapLevel": current_wrap_level,
                    "edgeIn": edge_in,
                    "edgeOut": edge_out
                })
                
            # Advance past connector
            while last_idx < len(pts) and pts[last_idx][1] <= conn_end_y:
                last_idx += 1
                
            if direction == 'R->L': # Exited right, entered left -> value crossed Vmax upward -> wrapLevel += 1
                current_wrap_level += 1
                wraps.append("wrap up")
            elif direction == 'L->R': # Exited left, entered right -> value crossed Vmin downward -> wrapLevel -= 1
                current_wrap_level -= 1
                wraps.append("wrap down")
                
        # Last fragment
        if last_idx < len(pts):
            frag_pts = pts[last_idx:]
            top_pt = frag_pts[0]
            bottom_pt = frag_pts[-1]
            edge_in = 'left' if abs(top_pt[0] - x_min) <= edge_tol else ('right' if abs(top_pt[0] - x_max) <= edge_tol else None)
            edge_out = 'left' if abs(bottom_pt[0] - x_min) <= edge_tol else ('right' if abs(bottom_pt[0] - x_max) <= edge_tol else None)
            
            fragments.append({
                "points": frag_pts,
                "wrapLevel": current_wrap_level,
                "edgeIn": edge_in,
                "edgeOut": edge_out
            })
            
        return {"fragments": fragments, "wraps": wraps}

    return guided_curve_track


if _HAS_FASTAPI:
    _register_endpoint()
