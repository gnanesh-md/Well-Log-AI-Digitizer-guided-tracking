import requests
import numpy as np
import cv2

# Create dummy image
img = np.ones((800, 600, 3), dtype=np.uint8) * 255
cv2.imwrite('dummy.jpg', img)

with open('dummy.jpg', 'rb') as f:
    resp = requests.post(
        'http://localhost:8123/segment-and-graph',
        files={'file': ('dummy.jpg', f, 'image/jpeg')},
        data={'include_header_ocr': True}
    )

print(resp.status_code)
print(resp.json().get('header_ocr'))
