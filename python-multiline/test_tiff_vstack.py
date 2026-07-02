import numpy as np

def test_vstack():
    f1 = np.zeros((100, 200, 3), dtype=np.uint8)
    f2 = np.zeros((150, 200, 3), dtype=np.uint8)
    print("Same width:", np.vstack([f1, f2]).shape)
    
    f3 = np.zeros((150, 210, 3), dtype=np.uint8)
    # This would fail: np.vstack([f1, f3])
    
    # Pad to max width
    frames = [f1, f3]
    max_w = max(f.shape[1] for f in frames)
    padded = []
    for f in frames:
        if f.shape[1] < max_w:
            pad_width = ((0, 0), (0, max_w - f.shape[1]), (0, 0))
            f = np.pad(f, pad_width, mode='constant', constant_values=255)
        padded.append(f)
    print("Padded shape:", np.vstack(padded).shape)

if __name__ == "__main__":
    test_vstack()
