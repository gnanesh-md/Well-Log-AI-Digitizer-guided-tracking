import numpy as np
import time

cols = 5
K = 6
A = np.random.rand(cols, K).astype(np.float32) * 10
c = 1.0

A1 = A.copy()
arg1 = np.tile(np.arange(K, dtype=np.int8), (cols, 1))

# Original loop
for k in range(1, K):
    better = A1[:, k - 1] + c < A1[:, k]
    A1[better, k] = A1[better, k - 1] + c
    arg1[better, k] = arg1[better, k - 1]

# Vectorized
A2 = A.copy()
arg2 = np.tile(np.arange(K, dtype=np.int8), (cols, 1))

B = A2 - c * np.arange(K)
# M is the running minimum
M = np.minimum.accumulate(B, axis=1)
# The index of the running minimum!
# Wait, how to get the argmin?
# We can use np.argmin? No, accumulate doesn't give arg.
