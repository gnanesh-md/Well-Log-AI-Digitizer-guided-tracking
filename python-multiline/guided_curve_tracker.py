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
from typing import List, Optional, Tuple

import cv2
import numpy as np

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
    sample_radius: int = 25,
) -> np.ndarray:
    """
    Learn what the target curve looks like (median LAB color) from small
    windows around the user's anchor points.

    User clicks are usually a few pixels off a 1-3 px wide curve, so the
    window is mostly paper. We therefore keep only pixels that genuinely
    stand out from the window's own background: inkness must exceed the
    window median by a clear margin, and we cap to the most inky pixels.
    """
    h, w = lab.shape[:2]
    samples = []
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
        ink = patch[mask]
        # cap to the most inky pixels so anti-aliased edges don't dominate
        if len(ink) > 40:
            order = np.argsort(-inkness[mask])
            ink = ink[order[:40]]
        samples.append(ink)
    if not samples:
        # fallback: pure black ink model
        return np.array([0.0, 128.0, 128.0], dtype=np.float32)
    allpix = np.concatenate(samples, axis=0)
    return np.median(allpix, axis=0).astype(np.float32)


def _estimate_curve_style(bgr: np.ndarray, curve_lab: np.ndarray, anchors: List[Tuple[int, int]]) -> bool:
    """
    Analyzes the physical length of the ink directly underneath the user's anchors.
    If the connected ink component is short (< 80px), it mathematically proves the curve is dashed.
    Returns: is_dashed (bool)
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_chroma = np.sqrt(curve_lab[1]**2 + curve_lab[2]**2)
    wL, wa, wb = (2.5, 1.0, 1.0) if target_chroma < 12 else (0.3, 2.5, 2.5)
    
    dL = lab[:, :, 0] - curve_lab[0]
    dL = np.where(dL < 0, dL * 0.8, dL)
    da = lab[:, :, 1] - curve_lab[1]
    db = lab[:, :, 2] - curve_lab[2]
    
    dist = np.sqrt(wL * dL * dL + wa * da * da + wb * db * db)
    cost = np.clip(dist / 60.0, 0.0, 1.0)
    
    ink_mask = (cost < 0.35).astype(np.uint8)
    
def _estimate_curve_style(bgr: np.ndarray, curve_lab: np.ndarray, anchors: List[Tuple[int, int]]) -> Tuple[bool, float]:
    """
    Returns (is_dashed, est_gap_px)
    """
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


def _build_cost_map(bgr: np.ndarray, curve_lab: np.ndarray, is_dashed: bool, suppress_gridlines: bool = True, est_gap_px: float = 15.0) -> np.ndarray:
    """
    Build a pixel-wise cost map for pathfinding.
    Cost is high for background/grid and low for pixels matching curve_lab.
    Adaptive Physics: Gap bridging and solid-line erasure is ONLY applied if is_dashed is true.
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

    # 1. ADAPTIVE SOLID LINE ERASURE
    if is_dashed:
        # Detect solid lines using connected components. A wavy solid line will be a single massive connected component.
        # A dashed line will be many tiny disconnected components.
        curve_ink = (cost < 0.35).astype(np.uint8)
        
    if suppress_gridlines:
        # The original code only suppressed dark gridlines. We safely extend it to suppress colored horizontal gridlines.
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        dark = (gray < 160).astype(np.uint8) * 255
        
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        colored_ink = (sat > 40).astype(np.uint8) * 255
        all_ink = cv2.bitwise_or(dark, colored_ink)
        
        # Horizontal lines: use all_ink, close dashed gridlines first, then detect > 51px straight
        hk_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        ink_closed_h = cv2.morphologyEx(all_ink, cv2.MORPH_CLOSE, hk_close)
        hk = cv2.getStructuringElement(cv2.MORPH_RECT, (51, 1))
        hlines = cv2.morphologyEx(ink_closed_h, cv2.MORPH_OPEN, hk)
        
        # Vertical lines: STRICTLY use dark ink only! Never erase vertical colored lines (which are the curves).
        vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 51))
        vlines = cv2.morphologyEx(dark, cv2.MORPH_OPEN, vk)
        
        grid = ((hlines > 0) | (vlines > 0))

        # Unconditionally penalize grid lines so they are strictly avoided
        cost = np.where(grid, np.minimum(cost + 0.8, 1.0), cost)

    # 2. ADAPTIVE GAP BRIDGING
    if is_dashed:
        # Erase massive solid lines to prevent jumping BEFORE computing the gap valleys
        _, core_ink = cv2.threshold((1.0 - cost).astype(np.float32) * 255, 127, 255, cv2.THRESH_BINARY)
        core_ink = core_ink.astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(core_ink, connectivity=8)
        
        solid_lines_mask = np.zeros_like(cost, dtype=bool)
        solid_threshold = max(100, 4 * est_gap_px)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_HEIGHT] > solid_threshold:
                solid_lines_mask[labels == i] = True
        
        # Soft Gaussian valley creates a smooth highway between dashed segments
        dash_mask = (cost < 0.45).astype(np.float32)
        # Wipe solid lines from the mask so they do NOT cast a valley
        dash_mask[solid_lines_mask] = 0.0
        
        # 2.2 Directional closings
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
            
        # 2.4 Hard guard: never let a bridge cross a column occupied by an erased solid line
        # This stops the bridge from "tunnelling" through the solid curve.
        union_mask[solid_lines_mask] = 0.0
            
        bg_mask = (1.0 - union_mask).astype(np.uint8)
        dist_trans = cv2.distanceTransform(bg_mask, cv2.DIST_L2, 3)
        
        # reach: dist_trans == 0 is 1.0, falls off to 0 over 15px
        reach = np.clip(1.0 - dist_trans / 15.0, 0.0, 1.0)
        
        # Thicken the solid line barrier to prevent DP from jumping over it
        thick_solid = cv2.dilate(solid_lines_mask.astype(np.uint8), np.ones((7, 7)))
        reach[thick_solid > 0] = 0.0
        
        cost = np.minimum(cost, np.clip(1.0 - reach * 10.0, 0.0, 1.0))
        
        # Unconditionally penalize the solid line pixels themselves
        cost = np.where(solid_lines_mask, 1.0, cost)
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

def _snap_anchor(cost: np.ndarray, x: int, y: int, radius: int = 25) -> Tuple[int, int]:
    """Snap a user click to the lowest-cost (most curve-like) pixel nearby."""
    h, w = cost.shape
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    win = cost[y0:y1, x0:x1].copy()
    # prefer pixels close to the click when costs tie
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
        return pts, 1.0, pts[len(pts)//2]

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
    if len(points) < window:
        return points
    xs = np.array([p[0] for p in points], dtype=np.float32)
    kernel = np.ones(window, dtype=np.float32) / window
    pad = window // 2
    xs_p = np.pad(xs, (pad, pad), mode="edge")
    xs_s = np.convolve(xs_p, kernel, mode="valid")
    return [[int(round(xs_s[i])), points[i][1]] for i in range(len(points))]


# --------------------------------------------------------------------------- #
#  Public tracing entry point
# --------------------------------------------------------------------------- #

def trace_guided_curve(
    bgr: np.ndarray,
    anchors_xy: List[Tuple[int, int]],
    snap_radius: int = 15,
    max_slope_px: int = 35,
    move_penalty: float = 0.02,
    curvature_penalty: float = 0.06,
    corridor_pad: int = 800,
    smooth_window: int = 7,
    suppress_gridlines: bool = True,
    curve_style: str = "auto",
    x_min: int = -1,
    x_max: int = -1,
) -> dict:
    if len(anchors_xy) < 2:
        raise ValueError("At least 2 anchor points are required")

    h, w = bgr.shape[:2]
    left_clip = max(0, x_min) if x_min >= 0 else 0
    right_clip = min(w - 1, x_max) if x_max >= 0 else (w - 1)
    anchors = [(int(np.clip(x, left_clip, right_clip)), int(np.clip(y, 0, h - 1)))
               for x, y in anchors_xy]

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    curve_lab = _sample_curve_appearance(lab, anchors)
    
    # 1. EXPLICIT STYLE OVERRIDE
    if curve_style in ("solid", "dashed"):
        is_dashed = (curve_style == "dashed")          # explicit user override wins
        style_source = "user"
        est_gap_px = 15.0 if is_dashed else 0.0
    else:                                               # "auto" (make this the new default)
        is_dashed, est_gap_px = _estimate_curve_style(bgr, curve_lab, anchors)
        style_source = "auto"
    
    # Store for JSON response
    detected_style = "dashed" if is_dashed else "solid"
    
    # Build cost map with adaptive gap bridging and solid line erasure
    cost = _build_cost_map(bgr, curve_lab, is_dashed, suppress_gridlines, est_gap_px)

    # 2. ADAPTIVE PHYSICS
    if is_dashed:
        curvature_penalty = 0.15  # Extremely stiff to blast through erased solid lines
    else:
        curvature_penalty = 0.05  # Highly flexible to follow tight bends

    snapped = [_snap_anchor(cost, x, y, snap_radius) for (x, y) in anchors]
    # Well-log curves are functions of depth: order anchors top -> bottom.
    snapped = sorted(snapped, key=lambda p: p[1])
    # Drop duplicates on the same row.
    dedup = [snapped[0]]
    for p in snapped[1:]:
        if p[1] != dedup[-1][1]:
            dedup.append(p)
    snapped = dedup
    if len(snapped) < 2:
        raise ValueError("Anchors collapsed to a single row - click points further apart vertically")

    all_points: List[List[int]] = []
    segments = []
    prev_exit_slope = -999.0
    for i in range(len(snapped) - 1):
        (xs_s, ys_s) = snapped[i]
        (xe_s, ye_s) = snapped[i + 1]
        dy = abs(ye_s - ys_s)
        dx = abs(xe_s - xs_s)
        
        if dy > 0:
            avg_slope = dx / dy
            seg_max_slope = min(120, int(max(max_slope_px, avg_slope * 1.2 + 2)))
            if is_dashed:
                seg_curvature_penalty = float(np.clip(curvature_penalty - avg_slope * 0.02, 0.03, curvature_penalty))
            else:
                seg_curvature_penalty = float(np.clip(curvature_penalty - avg_slope * 0.05, 0.01, curvature_penalty))
        else:
            seg_max_slope = max_slope_px
            seg_curvature_penalty = curvature_penalty * 0.75 if is_dashed else curvature_penalty * 0.25

        seg_pts, seg_cost, worst_pt = _trace_segment_dp(
            cost, snapped[i], snapped[i + 1],
            max_slope_px=seg_max_slope,
            move_penalty=move_penalty,
            curvature_penalty=seg_curvature_penalty,
            corridor_pad=corridor_pad,
            start_slope=prev_exit_slope if prev_exit_slope != -999.0 else None,
            endpoint_slope_weight=0.04 if prev_exit_slope != -999.0 else 0.0,
            x_min=x_min,
            x_max=x_max,
        )
        
        # 3.1 Compute exit slope
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
            seg_pts = seg_pts[1:]  # avoid duplicating the shared anchor row
        all_points.extend(seg_pts)
        
        seg_conf = float(np.clip(1.0 - seg_cost, 0.0, 1.0))
        # 7.1 "needs_anchor (confidence < ~0.75 OR the segment contains a detected crossing OR a bridged gap longer than 2 * est_gap_px)"
        # Note: bridged gap check is simplified to confidence < 0.75
        needs_anchor = seg_conf < 0.75
        
        segments.append({
            "from": list(snapped[i]),
            "to": list(snapped[i + 1]),
            "mean_cost": round(seg_cost, 4),
            "confidence": round(seg_conf, 4),
            "worst_pt": worst_pt,
            "needs_anchor": needs_anchor,
        })

    all_points = _smooth_x(all_points, smooth_window)
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
        file: UploadFile = File(...),
        points: str = Form(...),            # JSON: [[x, y], [x, y], ...] image-pixel coords
        snap_radius: int = Form(15),
        max_slope_px: int = Form(35),
        move_penalty: float = Form(0.02),
        curvature_penalty: float = Form(0.06),
        corridor_pad: int = Form(800),
        smooth_window: int = Form(7),
        suppress_gridlines: bool = Form(True),
        curve_style: str = Form("solid"),
        x_min: int = Form(-1),
        x_max: int = Form(-1),
    ):
        """
        Human-guided AI curve tracing.

        The frontend sends the displayed image plus the user's anchor clicks
        (in image pixel coordinates). Returns one traced point per depth row,
        passing exactly through every (snapped) anchor.
        """
        ext = (file.filename or "img.png").lower().rsplit(".", 1)[-1]
        if ext not in ("tif", "tiff", "png", "jpg", "jpeg", "bmp", "webp"):
            raise HTTPException(400, "Unsupported image format")

        data = await file.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise HTTPException(400, "Failed to decode image")

        try:
            anchors = json.loads(points)
            anchors = [(int(round(p[0])), int(round(p[1]))) for p in anchors]
        except Exception:
            raise HTTPException(400, "points must be JSON like [[x,y],[x,y],...]")

        if len(anchors) < 2:
            raise HTTPException(400, "At least 2 anchor points are required")

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
            )
        except ValueError as e:
            raise HTTPException(400, str(e))

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
