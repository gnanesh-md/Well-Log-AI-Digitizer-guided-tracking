import cv2
import numpy as np
from header_ocr_engine import extract_header_text_with_vllm, DEFAULT_MODEL

print("Model:", DEFAULT_MODEL)
img = np.zeros((1000, 1000, 3), dtype=np.uint8)
text, meta = extract_header_text_with_vllm(img, model_name=DEFAULT_MODEL)
print("Text:", text)
print("Meta:", meta)
