import numpy as np, cv2
from guided_curve_tracker import trace_guided_curve

def test_does_not_ride_vertical_gridline():
    H, W = 400, 300
    img = np.full((H, W, 3), (180, 205, 230), np.uint8)      # paper
    cv2.line(img, (150, 0), (150, H - 1), (40, 40, 40), 2)   # vertical gridline
    for yy in (80, 200, 320):
        cv2.line(img, (0, yy), (W - 1, yy), (60, 60, 60), 1) # horizontal rulings
    ys = np.arange(20, 381)
    xs = 150 - (90 * np.sin((ys - 20) / 360 * np.pi)).astype(int)  # dips to x~60
    pts = list(zip(xs.tolist(), ys.tolist()))
    for i in range(len(pts) - 1):
        cv2.line(img, pts[i], pts[i + 1], (30, 30, 30), 2)

    res = trace_guided_curve(img, [(150, 20), (150, 380)], curve_style="solid")
    trace_min_x = min(p[0] for p in res["points"])
    # Must follow the dip (min-x near 60), NOT ride the gridline (min-x near 150)
    assert trace_min_x < 90, f"tracer rode the gridline (min-x={trace_min_x})"

if __name__ == "__main__":
    test_does_not_ride_vertical_gridline()
    print("PASS: tracer follows the curve, not the gridline")
