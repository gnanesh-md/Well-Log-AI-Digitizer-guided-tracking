"""Regression test for human-guided dashed curve tracking next to solid and colored dotted curves."""

import cv2
import numpy as np
import sys
import os

from guided_curve_tracker import trace_guided_curve

def run_test():
    H, W = 900, 420
    # Paper-colored background
    img = np.full((H, W, 3), 245, np.uint8)

    # Gridlines
    for y in range(0, H, 60):
        cv2.line(img, (0, y), (W, y), (185, 185, 185), 1)
    for x in range(0, W, 50):
        cv2.line(img, (x, 0), (x, H), (185, 185, 185), 1)

    ys = np.arange(H)
    # Target black dashed curve
    xd = (210 + 70 * np.sin(ys / 70.0) + 18 * np.sin(ys / 19.0)).astype(int)
    # Orange dotted curve running 18px to the right of it
    xo = xd + 18
    # Black solid curve running 18px to the left of it
    xs = xd - 18

    for y in ys[:-1]:
        # Target black dashed curve
        if (y // 16) % 2 == 0:
            cv2.line(img, (int(xd[y]), int(y)), (int(xd[y + 1]), int(y + 1)), (20, 20, 20), 2)
        
        # Black solid curve
        cv2.line(img, (int(xs[y]), int(y)), (int(xs[y + 1]), int(y + 1)), (20, 20, 20), 2)

    # Orange dotted curve (dots drawn as small filled circles)
    for y in range(0, H - 1, 12):
        cv2.circle(img, (int(xo[y]), int(y)), 2, (30, 100, 240), -1)

    rng = np.random.default_rng(4)
    anchor_rows = [35, 160, 290, 430, 570, 710, 860]
    anchors = [
        (int(xd[r] + rng.integers(-4, 5)), int(r + rng.integers(-3, 4)))
        for r in anchor_rows
    ]

    res = trace_guided_curve(
        img,
        anchors,
        curve_style="dashed",
        corridor_pad=140,
        smooth_window=1,
    )
    pts = np.array(res["points"])
    err = np.array([abs(x - int(xd[y])) for x, y in pts if 0 <= y < H])

    print("--- Regression Test Results ---")
    print("Traced points:", len(pts))
    print("Mean |error| px:", round(float(err.mean()), 2))
    print("Median |error| px:", round(float(np.median(err)), 2))
    print("P95 |error| px:", round(float(np.percentile(err, 95)), 2))
    print("Max |error| px:", int(err.max()))
    print("Within 4px:", round(float((err <= 4).mean() * 100), 1), "%")
    
    # Check that color appearance is not hijacked by the orange curve.
    # The target curve color should be close to black/dark-gray: L should be low, 
    # and chroma (a, b distance from 128) should be very small.
    target_color = np.array(res["curve_color_lab"])
    chroma = abs(target_color[1] - 128) + abs(target_color[2] - 128)
    print(f"Estimated LAB color: {target_color}")
    print(f"Estimated chroma: {chroma}")

    # Save visualization for debugging
    vis = img.copy()
    for x, y in pts:
        vis[y, max(0, x - 1):x + 2] = (0, 200, 0)
    for x, y in res["snapped_anchors"]:
        cv2.circle(vis, (x, y), 5, (0, 0, 255), -1)
    cv2.imwrite("test_dashed_vs_colored_neighbor_trace.png", vis)
    print("Saved test_dashed_vs_colored_neighbor_trace.png")

    if np.median(err) > 4 or np.percentile(err, 95) > 25:
        raise SystemExit("Dashed tracker drifted too far from the dashed curve onto a neighbor curve!")

    if chroma > 10.0:
        raise SystemExit("Color appearance was hijacked by the orange dotted neighbor curve!")

    print("Test passed successfully!")

if __name__ == "__main__":
    run_test()
