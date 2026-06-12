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


def _build_cost_map(
    bgr: np.ndarray,
    curve_lab: np.ndarray,
    suppress_gridlines: bool = True,
) -> np.ndarray:
    """
    cost(pixel) in [0, 1]:  0 = looks exactly like the learned curve,
                            1 = looks nothing like it.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Highly prioritize color match (chroma) over lightness to track colored lines accurately even if they fade.
    # However, if the target curve is grayscale/black, we MUST rely heavily on lightness.
    target_chroma = abs(curve_lab[1] - 128) + abs(curve_lab[2] - 128)
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

    if suppress_gridlines:
        # Detect long straight horizontal & vertical strokes (grid).
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        dark = (gray < 160).astype(np.uint8) * 255
        hk = cv2.getStructuringElement(cv2.MORPH_RECT, (51, 1))
        vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 51))
        hlines = cv2.morphologyEx(dark, cv2.MORPH_OPEN, hk)
        vlines = cv2.morphologyEx(dark, cv2.MORPH_OPEN, vk)
        grid = ((hlines > 0) | (vlines > 0))

        # Unconditionally penalize grid lines so they are strictly avoided
        cost = np.where(grid, np.minimum(cost + 0.8, 1.0), cost)

    # Enhance dotted/dashed lines by closing vertical gaps in low-cost regions
    # Use a narrow vertical kernel to bridge gaps without bleeding horizontally into parallel curves
    curve_mask = (cost < 0.35).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 25))
    closed_mask = cv2.morphologyEx(curve_mask, cv2.MORPH_CLOSE, kernel)
    # Only fill gaps without destroying the deep sub-pixel valleys of the actual ink
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


def _trace_segment_dp(
    cost: np.ndarray,
    p_start: Tuple[int, int],
    p_end: Tuple[int, int],
    max_slope_px: int = 14,
    move_penalty: float = 0.030,
    curvature_penalty: float = 0.060,
    corridor_pad: int = 120,
) -> Tuple[List[List[int]], float]:
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
        return pts, c

    h, w = cost.shape
    cx0 = max(0, min(xs, xe) - corridor_pad)
    cx1 = min(w, max(xs, xe) + corridor_pad + 1)
    sub = cost[ys:ye + 1, cx0:cx1]
    rows, cols = sub.shape
    S = max_slope_px
    K = 2 * S + 1
    dxs = np.arange(-S, S + 1, dtype=np.int32)           # slope per state k
    move_pen = (move_penalty * np.abs(dxs)).astype(np.float32)

    INF = np.float32(1e9)
    dp = np.full((cols, K), INF, dtype=np.float32)
    dp[xs - cx0, :] = sub[0, xs - cx0]                   # start: any slope
    # parent slope-index chosen at each (row, x, k)
    parents = np.zeros((rows, cols, K), dtype=np.int8)

    for r in range(1, rows):
        # 1) min over previous slope j with L1 curvature penalty:
        #    A[x, k] = min_j dp[x, j] + curvature_penalty * |k - j|
        A = dp.copy()
        arg = np.tile(np.arange(K, dtype=np.int8), (cols, 1))
        for k in range(1, K):                            # forward pass
            better = A[:, k - 1] + curvature_penalty < A[:, k]
            A[better, k] = A[better, k - 1] + curvature_penalty
            arg[better, k] = arg[better, k - 1]
        for k in range(K - 2, -1, -1):                   # backward pass
            better = A[:, k + 1] + curvature_penalty < A[:, k]
            A[better, k] = A[better, k + 1] + curvature_penalty
            arg[better, k] = arg[better, k + 1]

        # 2) spatial shift by dx_k and add pixel + movement cost
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
        return pts, 1.0

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

    pts = [[int(px + cx0), int(ys + py)] for (px, py) in path_local]
    mean_cost = float(total / rows)
    return pts, mean_cost


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
    move_penalty: float = 0.04,
    curvature_penalty: float = 0.12,
    corridor_pad: int = 800,
    smooth_window: int = 7,
    suppress_gridlines: bool = True,
) -> dict:
    if len(anchors_xy) < 2:
        raise ValueError("At least 2 anchor points are required")

    h, w = bgr.shape[:2]
    anchors = [(int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1)))
               for x, y in anchors_xy]

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    curve_lab = _sample_curve_appearance(lab, anchors)
    cost = _build_cost_map(bgr, curve_lab, suppress_gridlines)

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
    for i in range(len(snapped) - 1):
        seg_pts, seg_cost = _trace_segment_dp(
            cost, snapped[i], snapped[i + 1],
            max_slope_px=max_slope_px,
            move_penalty=move_penalty,
            curvature_penalty=curvature_penalty,
            corridor_pad=corridor_pad,
        )
        if all_points:
            seg_pts = seg_pts[1:]  # avoid duplicating the shared anchor row
        all_points.extend(seg_pts)
        segments.append({
            "from": list(snapped[i]),
            "to": list(snapped[i + 1]),
            "mean_cost": round(seg_cost, 4),
            "confidence": round(float(np.clip(1.0 - seg_cost, 0.0, 1.0)), 4),
        })

    all_points = _smooth_x(all_points, smooth_window)
    overall_conf = float(np.mean([s["confidence"] for s in segments]))

    return {
        "points": all_points,                       # [[x, y], ...] one per row
        "snapped_anchors": [list(p) for p in snapped],
        "segments": segments,                       # per-segment confidence
        "confidence": round(overall_conf, 4),
        "curve_color_lab": [round(float(v), 1) for v in curve_lab],
        "num_points": len(all_points),
    }


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
        move_penalty: float = Form(0.04),
        curvature_penalty: float = Form(0.12),
        corridor_pad: int = Form(800),
        smooth_window: int = Form(7),
        suppress_gridlines: bool = Form(True),
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
            )
        except ValueError as e:
            raise HTTPException(400, str(e))

        return result

    return guided_curve_track


if _HAS_FASTAPI:
    _register_endpoint()
