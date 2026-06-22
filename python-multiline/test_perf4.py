import numpy as np
import time

cols = 360
K = 71
A = np.random.rand(cols, K).astype(np.float32) * 10
arg = np.tile(np.arange(K, dtype=np.int8), (cols, 1))

t0 = time.time()
for r in range(1000):
    for k in range(1, K):
        better = A[:, k - 1] + 0.1 < A[:, k]
        A[better, k] = A[better, k - 1] + 0.1
        arg[better, k] = arg[better, k - 1]
    for k in range(K - 2, -1, -1):
        better = A[:, k + 1] + 0.1 < A[:, k]
        A[better, k] = A[better, k + 1] + 0.1
        arg[better, k] = arg[better, k + 1]
t1 = time.time()

print(f"Forward/Backward passes: {t1-t0:.3f}s")
