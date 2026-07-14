import numpy as np
import cv2

H, W = 1000, 500
img = np.full((H, W, 3), (180, 205, 230), np.uint8)

for i in range(len(img)):
    if i % 100 == 0:
        cv2.line(img, (0, i), (W, i), (150, 150, 200), 1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
all_ink_mask = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10
)
print("all_ink_mask sum:", all_ink_mask.sum() / 255)
