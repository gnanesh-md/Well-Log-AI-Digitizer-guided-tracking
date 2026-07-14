import numpy as np
import time

cols = 360
K = 71
A = np.random.rand(cols, K).astype(np.float32) * 10
arg = np.tile(np.arange(K, dtype=np.int8), (cols, 1))
dxs = np.arange(-35, 36)
move_pen = np.abs(dxs) * 0.02
parents = np.zeros((6000, cols, K), dtype=np.int8)
INF = 1e9

t0 = time.time()
for r in range(1000):
    new_dp = np.full((cols, K), INF, dtype=np.float32)
    par_r = parents[r]
    for k in range(K):
        dx = dxs[k]
        if dx >= 0:
            src = slice(0, cols - dx) if dx > 0 else slice(0, cols)
            dst = slice(dx, cols) if dx > 0 else slice(0, cols)
        else:
            src = slice(-dx, cols)
            dst = slice(0, cols + dx)
        new_dp[dst, k] = A[src, k] + move_pen[k]
        par_r[dst, k] = arg[src, k]
t1 = time.time()

t2 = time.time()
for r in range(1000):
    new_dp = np.full((cols, K), INF, dtype=np.float32)
    par_r = parents[r]
    for k in range(K):
        dx = dxs[k]
        if dx > 0:
            new_dp[dx:, k] = A[:-dx, k] + move_pen[k]
            par_r[dx:, k] = arg[:-dx, k]
        elif dx < 0:
            new_dp[:dx, k] = A[-dx:, k] + move_pen[k]
            par_r[:dx, k] = arg[-dx:, k]
        else:
            new_dp[:, k] = A[:, k] + move_pen[k]
            par_r[:, k] = arg[:, k]
t3 = time.time()

print(f"Slice objects: {t1-t0:.3f}s")
print(f"Direct slices: {t3-t2:.3f}s")
