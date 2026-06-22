import numpy as np
import time

cols = 360
K = 71
A = np.random.rand(cols, K).astype(np.float32) * 10
arg = np.tile(np.arange(K, dtype=np.int8), (cols, 1))

A1 = A.copy()
arg1 = arg.copy()

t0 = time.time()
for _ in range(6000):
    for k in range(1, K):
        better = A1[:, k - 1] + 0.1 < A1[:, k]
        A1[better, k] = A1[better, k - 1] + 0.1
        arg1[better, k] = arg1[better, k - 1]
t1 = time.time()

A2 = A.copy()
arg2 = arg.copy()

t2 = time.time()
for _ in range(6000):
    for k in range(1, K):
        val = A2[:, k - 1] + 0.1
        better = val < A2[:, k]
        A2[:, k] = np.where(better, val, A2[:, k])
        arg2[:, k] = np.where(better, arg2[:, k - 1], arg2[:, k])
t3 = time.time()

print(f"Mask assignment: {t1-t0:.3f}s")
print(f"np.where: {t3-t2:.3f}s")
