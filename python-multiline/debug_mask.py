import numpy as np, cv2
from guided_curve_tracker import _build_cost_map
img = cv2.imread("test_dashed.png")
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
curve_lab = np.array([20, 128, 128])
cost = _build_cost_map(img, curve_lab, is_dashed=True)
cv2.imwrite("test_cost.png", (cost * 255).astype(np.uint8))
