import numpy as np
import time

cols = 360
K = 71
A = np.random.rand(K, cols).astype(np.float32) * 10
arg = np.tile(np.arange(K, dtype=np.int8)[:, None], (1, cols))

t0 = time.time()
for r in range(1000):
    for k in range(1, K):
        better = A[k - 1, :] + 0.1 < A[k, :]
        A[k, better] = A[k - 1, better] + 0.1
        arg[k, better] = arg[k - 1, better]
    for k in range(K - 2, -1, -1):
        better = A[k + 1, :] + 0.1 < A[k, :]
        A[k, better] = A[k + 1, better] + 0.1
        arg[k, better] = arg[k + 1, better]
t1 = time.time()

print(f"Transposed Forward/Backward passes: {t1-t0:.3f}s")
