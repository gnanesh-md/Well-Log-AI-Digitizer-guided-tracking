import requests

try:
    with open("tests/test_data/sample.tif", "wb") as f:
        f.write(b"dummy")
except:
    pass

# We don't have a real tif easily accessible from script, but we can send an empty file to see if it hits the endpoint at all.
files = {'file': ('dummy.tif', b'fake data', 'image/tiff')}
try:
    res = requests.post("http://127.0.0.1:8123/convert-image", files=files)
    print("Status Code:", res.status_code)
    print("Response:", res.text[:200])
except Exception as e:
    print("Error:", e)
