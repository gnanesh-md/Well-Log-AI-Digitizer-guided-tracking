import traceback
from fastapi.testclient import TestClient
from main import app
import cv2
import numpy as np
import json

client = TestClient(app)

img = np.full((500, 500, 3), 255, dtype=np.uint8)
cv2.line(img, (250, 0), (250, 500), (0, 0, 0), 2)
_, encoded = cv2.imencode(".png", img)

try:
    response = client.post(
        "/guided-curve-track",
        files={"file": ("test.png", encoded.tobytes(), "image/png")},
        data={
            "points": json.dumps([[250, 100], [250, 400]]),
            "x_min": "-1",
            "x_max": "-1"
        }
    )
    print("STATUS:", response.status_code)
    print("RESPONSE:", response.text)
except Exception as e:
    print("Exception!")
    traceback.print_exc()
