import numpy as np, cv2
from guided_curve_tracker import trace_guided_curve

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

def evaluate_tracking(pts, true_x):
    traced_x_arr = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    
    # ensure ys match true_x indices
    valid = (ys >= 0) & (ys < len(true_x))
    traced_x_arr = traced_x_arr[valid]
    ys = ys[valid]
    
    errors = np.abs(traced_x_arr - true_x[ys])
    mean_err = np.mean(errors)
    within_4px = np.mean(errors <= 4)
    # Wrong curve is roughly error > 100 near a crossing
    return mean_err, within_4px

print("Building 55px gap dashed curve...")
img, x_solid, x_dashed = build_log(dash_on=15, dash_off=55)

# Place anchors far apart
anchors = [(int(x_dashed[y]), y) for y in range(100, 901, 200)]

print("Tracing guided curve with auto style...")
res = trace_guided_curve(img, anchors, curve_style="auto")
print("Detected Style:", res.get("detected_style"))

mean_err, within_4px = evaluate_tracking(res["points"], x_dashed)
print(f"Mean error: {mean_err:.2f} px")
print(f"Within 4px: {within_4px:.1%}")

print("Segments needing anchors:", [s["needs_anchor"] for s in res["segments"]])
