import math
from typing import Callable, Tuple

import numpy as np
from scipy.linalg import expm, eig
import matplotlib.pyplot as plt

# 
# # -------------------------
# # Helpers
# # -------------------------
# def dominant_real_eigvec(A):
#     w, v = eig(A)
#     idx = int(np.argmax(np.real(w)))
#     vec = np.real(v[:, idx])
#     if np.linalg.norm(vec) < 1e-16:
#         vec = np.zeros(A.shape[0])
#         vec[0] = 1.0
#     return vec / np.linalg.norm(vec)
# 
# 
# # -------------------------
# # Curve generation (iterative)
# # -------------------------
# def generate_curve_iter(A, b, T=40.0, dt=0.05, tol=1e-8, max_steps=10_000_00):
#     expAdt = expm(A * dt)
#     x = b.astype(float).copy()
#     X = [x.copy()]
#     t_vals = [0.0]
#     prev = x.copy()
#     steps = min(int(np.ceil(T / dt)), max_steps)
#     for i in range(1, steps + 1):
#         x = expAdt @ x
#         X.append(x.copy())
#         t_vals.append(i * dt)
#         if np.linalg.norm(x - prev) < tol:
#             # early stop
#             break
#         prev = x.copy()
#     return np.array(t_vals), np.vstack(X)
# 
# 
# # -------------------------
# # Corrected 2D picker:
# #   - fixed index mapping bug
# #   - choose radial direction if point not near zero
# # -------------------------
# def pick_two_support_rays_2d(A, X, min_norm_ratio=1e-6, eps_tangent=1e-12):
#     """
#     X: (m,2) curve samples.
#     Returns: rays (2x2 rows unit), idxs (2,), arc_len_rad, info
#     """
#     norms = np.linalg.norm(X, axis=1)
#     if X.shape[1] != 2:
#         raise ValueError("2D function expects 2D points")
# 
#     max_norm = norms.max() if norms.size else 0.0
#     if max_norm == 0:
#         # trivial fallback
#         v = dominant_real_eigvec(A)
#         perp = np.array([-v[1], v[0]])
#         return np.vstack([v / np.linalg.norm(v), perp / np.linalg.norm(perp)]), np.array([0, 0]), np.pi, {
#             "reason": "all zero"}
# 
#     thr = max_norm * min_norm_ratio
#     valid = np.where(norms >= thr)[0]
#     if valid.size == 0:
#         v = dominant_real_eigvec(A)
#         perp = np.array([-v[1], v[0]])
#         return np.vstack([v / np.linalg.norm(v), perp / np.linalg.norm(perp)]), np.array([0, 0]), np.pi, {
#             "reason": "no valid points"}
# 
#     thetas = np.arctan2(X[valid, 1], X[valid, 0])
#     ang_sorted_idx = np.argsort(thetas)
#     ang_sorted = thetas[ang_sorted_idx]
#     # circular gaps
#     gaps = np.diff(np.concatenate([ang_sorted, ang_sorted[:1] + 2 * np.pi]))
#     k = int(np.argmax(gaps))
#     largest_gap = gaps[k]
#     arc_len = 2 * np.pi - largest_gap
#     a_start = ang_sorted[(k + 1) % ang_sorted.size]
#     a_end = ang_sorted[k]
#     if a_end < a_start:
#         a_end += 2 * np.pi
# 
#     # NOTE: here we must map back to original indices correctly (bug fixed)
#     thetas_shifted = np.mod(thetas - a_start + 2 * np.pi, 2 * np.pi)
#     idx_rel_start = int(np.argmin(np.abs(thetas_shifted - 0.0)))  # index into `thetas` (and into valid)
#     idx_rel_end = int(np.argmin(np.abs(thetas_shifted - arc_len)))  # index into `thetas`
#     idx_start = valid[idx_rel_start]  # FIXED mapping: use valid[...] directly
#     idx_end = valid[idx_rel_end]
# 
#     pos_start = X[idx_start].copy()
#     pos_end = X[idx_end].copy()
# 
#     # choose ray for endpoint: radial if point not tiny; otherwise tangent or eigenvector
#     def choose_ray(A, pos, max_norm, pos_thr_ratio=1e-6, eps_tan=1e-12):
#         pos_norm = np.linalg.norm(pos)
#         pos_thr = max_norm * pos_thr_ratio
#         if pos_norm >= pos_thr:
#             ray = pos / pos_norm  # radial through the actual point
#             reason = "radial"
#         else:
#             tang = A @ pos
#             if np.linalg.norm(tang) >= eps_tan:
#                 ray = tang / np.linalg.norm(tang)
#                 reason = "tangent"
#             else:
#                 ray = dominant_real_eigvec(A)
#                 reason = "eigvec_fallback"
#         # ensure ray points 'outwards' relative to pos (if pos not tiny)
#         if pos_norm > 1e-12 and np.dot(ray, pos) < 0:
#             ray = -ray
#         return ray, reason
# 
#     r1, reason1 = choose_ray(A, pos_start, max_norm, pos_thr_ratio=min_norm_ratio, eps_tan=eps_tangent)
#     r2, reason2 = choose_ray(A, pos_end, max_norm, pos_thr_ratio=min_norm_ratio, eps_tan=eps_tangent)
# 
#     rays = np.vstack([r1, r2])
#     info = {"idx_start": int(idx_start), "idx_end": int(idx_end),
#             "reason_start": reason1, "reason_end": reason2}
#     return rays, np.array([idx_start, idx_end]), arc_len, info


# -------------------------
# Plot 2D
# -------------------------
def plot_2d_with_rays(X, rays, A=None, show_eigvecs=False, title="2D curve and support rays"):
    plt.figure(figsize=(6, 6))
    plt.plot(X[:, 0], X[:, 1], '-k', lw=1.2, label='curve x(t)')
    max_r = np.max(np.linalg.norm(X, axis=1)) * 1.05
    for i, r in enumerate(rays):
        plt.plot([0, r[0] * max_r], [0, r[1] * max_r], '--', lw=2, label=f'ray {i + 1}')
    if show_eigvecs and A is not None:
        w, v = eig(A)
        for i in range(v.shape[1]):
            vec = np.real(v[:, i])
            if np.linalg.norm(vec) < 1e-12:
                continue
            vec = vec / np.linalg.norm(vec)
            plt.plot([0, vec[0] * max_r], [0, vec[1] * max_r], ':', linewidth=2, color='red', label=f"eigvec {i + 1}")
    plt.gca().set_aspect('equal', 'box')
    plt.grid(True)
    plt.legend()
    plt.title(title)
    plt.show()


# -------------------------
# Example run (your matrix)
# -------------------------
# if __name__ == "__main__":
#     A = np.array([[0.0, 1.0],
#                   [-0.1, -0.7]])
#     b = np.array([0.0, 1.5])
# 
#     t, X = generate_curve_iter(A, b, T=80.0, dt=0.05, tol=1e-9)
# 
#     rays, idxs, arc_len, info = pick_two_support_rays_2d(A, X, min_norm_ratio=1e-6)
#     print("Selected sample indices:", idxs)
#     print("Arc length (rad):", arc_len)
#     print("Selection info:", info)
#     print("Ray directions (unit rows):\n", rays)
# 
#     # Check that one of rays goes through the starting point (0,1.5)
#     # The radial ray for the starting point should be [0, +1] (up to sign)
#     print("Start sample point:", X[idxs[0]])
#     print("End sample point:  ", X[idxs[1]])
# 
#     plot_2d_with_rays(X, rays, A=A, show_eigvecs=True,
#                       title="Исправлено: луч через начальную точку + поддержка для нулевого конца")


def is_hurwitz_matrix(A):
    w, _ = eig(A)
    return np.all(np.real(w) < 0)


def generate_curve_iter(A, b, T=40.0, dt=0.05, tol=1e-8, max_steps=10_000_00):
    if not is_hurwitz_matrix(A):
        raise ValueError("A is not a Hurwitz matrix")
    expAdt = expm(A * dt)
    x = b.astype(float).copy()
    X = [x.copy()]
    prev = x.copy()
    steps = min(int(np.ceil(T / dt)), max_steps)
    for i in range(1, steps + 1):
        x = expAdt @ x
        X.append(x.copy())
        if np.linalg.norm(x - prev) < tol:
            # early stop
            break
        prev = x.copy()
    return np.vstack(X)


def tangent_from_table(data: np.ndarray, idx: int) -> Tuple[Callable[[float], float], float, float]:
    """
    Находит касательную к таблично заданной функции в точке data[idx].

    Параметры:
        data: np.ndarray формы (n, 2), где каждая строка = [x, y].
        idx: индекс строки, в которой требуется касательная.

    Возвращает:
        (tangent_func, slope m, intercept b),
        где tangent_func(x) = m * x + b.
    """
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("Ожидается массив формы (n, 2)")

    n = data.shape[0]
    # if not (0 <= idx < n):
    #     raise IndexError("Индекс вне диапазона")

    x0, y0 = data[idx]

    # Вычисляем наклон: если возможно, используем центральную разность
    if 0 < idx < n - 1:
        x_prev, y_prev = data[idx - 1]
        x_next, y_next = data[idx + 1]
        m = (y_next - y_prev) / (x_next - x_prev)
    elif idx == 0:  # край слева — вперёд
        x1, y1 = data[idx + 1]
        m = (y1 - y0) / (x1 - x0)
    else:  # край справа — назад
        x1, y1 = data[idx - 1]
        m = (y0 - y1) / (x0 - x1)

    b = y0 - m * x0

    def tangent(x: float) -> float:
        return m * x + b

    return tangent, float(m), float(b)


def pick_two_support_rays_2d(A, X):
    r1 = b.copy()

    tan_func, m, _ = tangent_from_table(X, -1)
    x = -1 / m
    y = tan_func(x)

    r2 = np.array([x, y])
    
    r1 = r1 / np.linalg.norm(r1)
    r2 = r2 / np.linalg.norm(r2)
    
    return np.vstack([r1, r2])


if __name__ == "__main__":
    A = np.array([[0.0, 1.0],
                  [-0.1, -0.7]])
    b = np.array([0.0, 1.5])

    X = generate_curve_iter(A, b, T=80.0, dt=0.05, tol=1e-9)

    rays = pick_two_support_rays_2d(A, X)
    # print("Selected sample indices:", idxs)
    # print("Arc length (rad):", arc_len)
    # print("Selection info:", info)
    print("Ray directions (unit rows):\n", rays)
    # 
    # # Check that one of rays goes through the starting point (0,1.5)
    # # The radial ray for the starting point should be [0, +1] (up to sign)
    # print("Start sample point:", X[idxs[0]])
    # print("End sample point:  ", X[idxs[1]])

    plot_2d_with_rays(X, rays, A=A, show_eigvecs=False)
