# """
# exp_cone_fixed_v2.py
# 
# Исправлённая версия: гарантированный seed-подбор лучей по угловому покрытию,
# далее переход к NNLS; защита от слишком строгого tol.
# Требования: numpy, scipy, matplotlib
# """
# 
# import numpy as np
# from scipy.linalg import expm
# from scipy.optimize import nnls
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# 
# 
# # ---------------------------
# # Helpers
# # ---------------------------
# 
# def safe_normalize_rows(X, eps=1e-12):
#     norms = np.linalg.norm(X, axis=1, keepdims=True)
#     norms_safe = np.where(np.isfinite(norms) & (norms > eps), norms, eps)
#     U = X / norms_safe
#     U = np.nan_to_num(U, nan=0.0, posinf=0.0, neginf=0.0)
#     return U, norms.ravel()
# 
# 
# def sample_curve(A, b, T=10.0, num=1001):
#     t = np.linspace(0.0, T, num)
#     X = np.stack([expm(A * tt) @ b for tt in t], axis=0)
#     return t, X
# 
# 
# # ---------------------------
# # Incremental clustering (unchanged)
# # ---------------------------
# 
# def cluster_directions_incremental(unit_dirs,
#                                    angular_tol_deg=5.0,
#                                    max_clusters=200,
#                                    subsample_for_clustering=None,
#                                    verbose=False):
#     N, n = unit_dirs.shape
#     if N == 0:
#         return np.zeros((0, n)), np.zeros(0, dtype=int)
#     theta = np.deg2rad(max(0.0, min(179.9, angular_tol_deg)))
#     cos_thresh = np.cos(theta)
#     if subsample_for_clustering is not None and N > subsample_for_clustering:
#         rng = np.random.default_rng(0)
#         subs_idx = rng.choice(N, size=subsample_for_clustering, replace=False)
#         order = list(subs_idx) + [i for i in range(N) if i not in set(subs_idx)]
#     else:
#         order = list(range(N))
#     sums = []
#     counts = []
#     labels = -np.ones(N, dtype=int)
#     created_clusters = 0
#     for pos, idx in enumerate(order):
#         u = unit_dirs[idx]
#         if created_clusters == 0:
#             sums.append(u.copy());
#             counts.append(1);
#             labels[idx] = 0;
#             created_clusters = 1
#             continue
#         centroids = np.vstack(sums)
#         cnorms = np.linalg.norm(centroids, axis=1, keepdims=True)
#         cnorms = np.maximum(cnorms, 1e-12)
#         centroids_normed = centroids / cnorms
#         dots = centroids_normed @ u
#         best = np.argmax(dots)
#         best_dot = dots[best]
#         if best_dot >= cos_thresh and created_clusters < max_clusters:
#             sums[best] += u;
#             counts[best] += 1;
#             labels[idx] = best
#         else:
#             if created_clusters < max_clusters:
#                 sums.append(u.copy());
#                 counts.append(1);
#                 labels[idx] = created_clusters;
#                 created_clusters += 1
#             else:
#                 labels[idx] = best;
#                 sums[best] += u;
#                 counts[best] += 1
#     sums_arr = np.vstack(sums)
#     norms = np.linalg.norm(sums_arr, axis=1, keepdims=True)
#     norms = np.maximum(norms, 1e-12)
#     candidates = sums_arr / norms
#     if verbose:
#         print(
#             f"[cluster-inc] angular_tol_deg={angular_tol_deg}, produced {candidates.shape[0]} clusters (max_clusters={max_clusters})")
#     return candidates, labels
# 
# 
# # ---------------------------
# # safe NNLS residuals
# # ---------------------------
# 
# def safe_nnls_residuals(unit_dirs, rays, idxs=None):
#     if rays.shape[0] == 0:
#         if idxs is None:
#             return np.full(unit_dirs.shape[0], np.inf)
#         else:
#             return np.full(len(idxs), np.inf)
#     if not np.all(np.isfinite(rays)):
#         if idxs is None:
#             return np.full(unit_dirs.shape[0], np.inf)
#         else:
#             return np.full(len(idxs), np.inf)
#     R = rays.T
#     if idxs is None:
#         idxs = np.arange(unit_dirs.shape[0])
#     out = np.empty(len(idxs), dtype=float)
#     for i, idx in enumerate(idxs):
#         u = unit_dirs[idx]
#         try:
#             coeffs, _ = nnls(R, u)
#             out[i] = np.linalg.norm(R @ coeffs - u)
#         except Exception:
#             out[i] = np.inf
#     return out
# 
# 
# def prune_redundant_rays(rays, tol=1e-9, verbose=False):
#     k, n = rays.shape
#     keep_mask = np.ones(k, dtype=bool)
#     for j in range(k):
#         if not keep_mask[j]:
#             continue
#         others = [i for i in range(k) if i != j and keep_mask[i]]
#         if len(others) == 0:
#             continue
#         R = rays[others].T
#         target = rays[j]
#         try:
#             c, _ = nnls(R, target)
#             resid = np.linalg.norm(R @ c - target)
#         except Exception:
#             resid = np.inf
#         if resid <= tol:
#             keep_mask[j] = False
#             if verbose:
#                 print(f"[prune] remove ray {j} (resid={resid:.2e})")
#     kept = np.where(keep_mask)[0]
#     return rays[kept], kept
# 
# 
# # ---------------------------
# # Greedy selection with seed rule (fixed)
# # ---------------------------
# 
# def greedy_cover_select_rays_on_candidates(unit_dirs, candidate_rays,
#                                            angular_tol_deg=6.0,
#                                            tol=1e-6,
#                                            max_k=None,
#                                            cover_fraction=1.0,
#                                            verbose=True):
#     """
#     Исправлённый greedy:
#       - сначала выбираем seed по угловому критерию (dot >= cos(seed_angle))
#         чтобы гарантировать, что первый кандидат покрывает группу точек;
#       - далее используем NNLS-остатки для точного покрытия и обновления covered.
#     """
#     N = unit_dirs.shape[0]
#     M = candidate_rays.shape[0]
#     if M == 0:
#         return np.empty((0, unit_dirs.shape[1])), np.array([], dtype=int), np.full(N, np.inf)
#     if max_k is None:
#         max_k = M
# 
#     # precompute dot products N x M
#     D = unit_dirs @ candidate_rays.T  # shape (N, M)
# 
#     # determine seed angular threshold (we make seed angle slightly larger than cluster angle)
#     # seed_angle = max(angular_tol_deg * 1.2, 10 degrees) capped at 60 deg
#     seed_angle = float(max(angular_tol_deg * 1.2, 10.0))
#     seed_angle = min(seed_angle, 60.0)
#     cos_seed = np.cos(np.deg2rad(seed_angle))
#     # convert to residual corresponding to single-ray projection: resid = sqrt(1 - (dot)^2)
#     tol_from_seed = np.sqrt(max(0.0, 1.0 - cos_seed * cos_seed))
#     # effective tol cannot be smaller than that (otherwise single-ray coverage by dot>=cos_seed is impossible)
#     tol_eff = max(tol, tol_from_seed)
# 
#     if verbose:
#         print(
#             f"[greedy-cand-fixed] seed_angle={seed_angle:.2f}deg, cos_seed={cos_seed:.4f}, tol_from_seed={tol_from_seed:.3e}, tol_eff={tol_eff:.3e}")
# 
#     covered = np.zeros(N, dtype=bool)
#     selected_idx = []
#     selected_rays = []
#     uncovered_idxs = np.where(~covered)[0]
#     iter_no = 0
# 
#     while True:
#         iter_no += 1
#         if verbose:
#             print(
#                 f"[greedy-cand-fixed] iter {iter_no}: uncovered {uncovered_idxs.size}/{N}, selected {len(selected_idx)}")
#         if uncovered_idxs.size == 0:
#             break
#         if len(selected_idx) >= max_k:
#             if verbose:
#                 print("[greedy-cand-fixed] reached max_k")
#             break
# 
#         best_new = -1
#         best_j = None
# 
#         # seed-phase: if no selected rays yet, choose candidate maximizing angular coverage (D >= cos_seed)
#         if len(selected_idx) == 0:
#             # count how many uncovered points have dot >= cos_seed for each candidate
#             counts = np.count_nonzero(D[uncovered_idxs, :] >= cos_seed, axis=0)
#             best_j = int(np.argmax(counts))
#             best_new = int(counts[best_j])
#             if best_new <= 0:
#                 # fallback: choose candidate maximizing sum of positive dots (closest hemisphere)
#                 sums_pos = np.sum(np.clip(D[uncovered_idxs, :], 0.0, None), axis=0)
#                 best_j = int(np.argmax(sums_pos))
#                 best_new = int(np.count_nonzero(D[uncovered_idxs, best_j] > 0.0))
#             if verbose:
#                 print(f"[greedy-cand-fixed] seed chosen candidate {best_j}, angular-covered={best_new}")
#         else:
#             # general phase: test each candidate by NNLS residuals on uncovered set
#             for j in range(M):
#                 if j in selected_idx:
#                     continue
#                 if len(selected_rays) == 0:
#                     R_try = candidate_rays[[j]]
#                 else:
#                     R_try = np.vstack([np.array(selected_rays), candidate_rays[[j]]])
#                 resid = safe_nnls_residuals(unit_dirs, R_try, idxs=uncovered_idxs)
#                 new_cover = int(np.count_nonzero(resid <= tol_eff))
#                 if new_cover > best_new:
#                     best_new = new_cover
#                     best_j = j
# 
#         if best_j is None or best_new <= 0:
#             if verbose:
#                 print("[greedy-cand-fixed] no candidate improves coverage -> stopping")
#             break
# 
#         # add best_j
#         selected_idx.append(best_j)
#         selected_rays.append(candidate_rays[best_j])
#         R_current = np.vstack(selected_rays)
#         resid_current = safe_nnls_residuals(unit_dirs, R_current)
#         covered = covered | (resid_current <= tol_eff)
#         uncovered_idxs = np.where(~covered)[0]
#         covered_fraction = covered.sum() / float(N)
#         if verbose:
#             print(
#                 f"[greedy-cand-fixed] added candidate {best_j}, newly covered={best_new}, covered_fraction={covered_fraction:.6f}")
#         if covered_fraction >= cover_fraction:
#             if verbose:
#                 print("[greedy-cand-fixed] desired cover_fraction reached")
#             break
# 
#     if len(selected_rays) == 0:
#         return np.empty((0, unit_dirs.shape[1])), np.array([], dtype=int), np.full(N, np.inf)
# 
#     selected_rays_arr = np.vstack(selected_rays)
#     pruned, kept_local = prune_redundant_rays(selected_rays_arr, tol=tol_eff * 0.1, verbose=verbose)
#     selected_candidate_indices = np.array(selected_idx)[kept_local]
#     final_resid = safe_nnls_residuals(unit_dirs, pruned)
#     return pruned, selected_candidate_indices, final_resid
# 
# 
# # ---------------------------
# # High-level wrapper (calls clustering + greedy)
# # ---------------------------
# 
# def analyze_and_select_cone(A, b, T=10.0, num=1201,
#                             angular_tol_deg=5.0, max_clusters=200,
#                             subsample_for_clustering=4000,
#                             tol=1e-6, max_k_factor=4,
#                             cover_fraction=1.0, verbose=True):
#     t, X = sample_curve(A, b, T=T, num=num)
#     unit_dirs, norms = safe_normalize_rows(X)
#     n = X.shape[1]
#     candidates, labels = cluster_directions_incremental(unit_dirs,
#                                                         angular_tol_deg=angular_tol_deg,
#                                                         max_clusters=max_clusters,
#                                                         subsample_for_clustering=subsample_for_clustering,
#                                                         verbose=verbose)
#     candidates, _ = safe_normalize_rows(np.atleast_2d(candidates))
#     if verbose:
#         print(f"[analyze] sampled {X.shape[0]} points, produced {candidates.shape[0]} candidates after clustering")
#     max_k = max(1, int(max_k_factor * n))
#     selected_rays, sel_candidate_indices, resid_all = greedy_cover_select_rays_on_candidates(
#         unit_dirs, candidates,
#         angular_tol_deg=angular_tol_deg,
#         tol=tol,
#         max_k=max_k,
#         cover_fraction=cover_fraction,
#         verbose=verbose
#     )
#     # representative sample indices for each selected candidate (first member of cluster) - optional
#     sel_sample_indices = []
#     for cand_idx in sel_candidate_indices:
#         ids = np.where(labels == cand_idx)[0]
#         sel_sample_indices.append(int(ids[0]) if ids.size > 0 else -1)
#     return {
#         "t": t,
#         "X": X,
#         "unit_dirs": unit_dirs,
#         "norms": norms,
#         "candidates": candidates,
#         "cluster_labels": labels,
#         "selected_rays": selected_rays,
#         "selected_candidate_indices": sel_candidate_indices,
#         "selected_sample_indices": np.array(sel_sample_indices, dtype=int),
#         "resid_all": resid_all
#     }
# 
# 
# # ---------------------------
# # Visualization
# # ---------------------------
# 
# def plot_2d_curve_and_cone(X, rays, title="2D: curve and cone"):
#     plt.figure(figsize=(6, 6))
#     plt.plot(X[:, 0], X[:, 1], '-', lw=2, label='curve x(t)')
#     if rays.shape[0] > 0:
#         max_norm = np.max(np.linalg.norm(X, axis=1))
#         scaled = rays * max_norm * 1.05
#         for i, r in enumerate(scaled):
#             plt.plot([0, r[0]], [0, r[1]], '--', lw=2, label=f'ray {i + 1}')
#         if rays.shape[0] == 2:
#             poly_x = [0, scaled[0, 0], scaled[1, 0]]
#             poly_y = [0, scaled[0, 1], scaled[1, 1]]
#             plt.fill(poly_x, poly_y, alpha=0.2)
#     plt.gca().set_aspect('equal', 'box')
#     plt.grid(True)
#     plt.legend()
#     plt.title(title)
#     plt.xlabel('x1');
#     plt.ylabel('x2')
#     plt.show()
# 
# 
# def plot_3d_curve_and_cone(X, rays, title="3D: curve and cone"):
#     fig = plt.figure(figsize=(9, 7))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot(X[:, 0], X[:, 1], X[:, 2], '-', lw=2, label='curve x(t)')
#     if rays.shape[0] > 0:
#         max_norm = np.max(np.linalg.norm(X, axis=1))
#         scaled = rays * max_norm * 1.05
#         for i, r in enumerate(scaled):
#             ax.plot([0, r[0]], [0, r[1]], [0, r[2]], '--', lw=2, label=f'ray {i + 1}')
#         if rays.shape[0] == 3:
#             faces = [[[0, 0, 0], scaled[0], scaled[1]],
#                      [[0, 0, 0], scaled[1], scaled[2]],
#                      [[0, 0, 0], scaled[2], scaled[0]]]
#             poly = Poly3DCollection(faces, alpha=0.15);
#             ax.add_collection3d(poly)
#     all_pts = np.vstack([X] + ([rays * np.max(np.linalg.norm(X, axis=1)) * 1.05] if rays.shape[0] > 0 else []))
#     mins = all_pts.min(axis=0);
#     maxs = all_pts.max(axis=0)
#     ax.set_xlim(mins[0] * 1.1, maxs[0] * 1.1);
#     ax.set_ylim(mins[1] * 1.1, maxs[1] * 1.1);
#     ax.set_zlim(mins[2] * 1.1, maxs[2] * 1.1)
#     ax.set_box_aspect([1, 1, 1]);
#     ax.legend();
#     ax.set_title(title)
#     plt.show()
# 
# 
# # ---------------------------
# # Examples
# # ---------------------------
# 
# def run_example_2d():
#     A = np.array([[-1.0, 0.8], [0.0, -2.0]])
#     b = np.array([1.0, 0.6])
# 
#     A2 = np.array([
#         [0.0, 1.0],
#         [-0.1, -0.7]
#     ])
#     b2 = np.array([0, 1.5])
# 
#     # result = analyze_and_select_cone(A2, b2,
#     #                                  T=3000.0, num=15001,
#     #                                  angular_tol_deg=6.0, max_clusters=80,
#     #                                  tol=1e-6, max_k_factor=4,
#     #                                  cover_fraction=1.0, verbose=True)
#     
#     # large num supported via subsample_for_clustering
#     res = analyze_and_select_cone(A2, b2, T=2000.0, num=12001,
#                                   angular_tol_deg=6.0, max_clusters=80,
#                                   subsample_for_clustering=1000,
#                                   tol=1e-6, max_k_factor=4, cover_fraction=1.0, verbose=True)
#     print("2D selected rays:", res["selected_rays"].shape[0], "max resid:", np.nanmax(res["resid_all"]))
#     plot_2d_curve_and_cone(res["X"], res["selected_rays"], title=f"2D cone ({res['selected_rays'].shape[0]} rays)")
# 
# 
# def run_example_3d():
#     D = np.diag([-0.6, -1.2, -2.0])
#     Q = np.array([[1.0, 0.3, 0.2], [0.2, 1.0, 0.4], [0.1, 0.2, 1.0]])
#     A = np.linalg.inv(Q) @ D @ Q
#     b = np.array([1.0, 0.5, 0.2])
#     res = analyze_and_select_cone(A, b, T=100.0, num=3001,
#                                   angular_tol_deg=6.0, max_clusters=120,
#                                   subsample_for_clustering=1500,
#                                   tol=1e-6, max_k_factor=4, cover_fraction=1.0, verbose=True)
#     print("3D selected rays:", res["selected_rays"].shape[0], "max resid:", np.nanmax(res["resid_all"]))
#     plot_3d_curve_and_cone(res["X"], res["selected_rays"], title=f"3D cone ({res['selected_rays'].shape[0]} rays)")
# 
# 
# if __name__ == "__main__":
#     curve = np.loadtxt("D:/Programming/C++/preisach_hysteresis_model/PreisachModel/curve2d.csv", delimiter=",", skiprows=0)
#     rays = np.loadtxt("D:/Programming/C++/preisach_hysteresis_model/PreisachModel/rays2d.csv", delimiter=",", skiprows=0)
#     plt.plot(curve[:, 0], curve[:, 1], 'b-', label="Curve")
#     origin = np.zeros((rays.shape[0], 2))  # массив нулей (M,2)
#     plt.quiver(origin[:, 0], origin[:, 1], rays[:, 0], rays[:, 1],
#                angles='xy', scale_units='xy', scale=1, color='r', label="Rays")
#     plt.xlabel("x")
#     plt.ylabel("y")
#     plt.axis("equal")
#     plt.legend()
#     plt.grid(True)
#     plt.show()
#     
#     # run_example_2d()
#     # run_example_3d()


import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

def sample_curve(A, b, T=10.0, num=1001):
    t = np.linspace(0, T, num)
    X = np.stack([expm(A * tt) @ b for tt in t], axis=0)
    return t, X

def two_rays_by_angle(X, eps=1e-12):
    # X: (m,2) array of points
    # remove (or warn on) points near origin
    norms = np.linalg.norm(X, axis=1)
    if np.any(norms < eps):
        print("Кривая проходит слишком близко к нулю — двух лучей может быть недостаточно.")
    thetas = np.arctan2(X[:,1], X[:,0])  # in (-pi, pi]
    # unwrap angles to avoid artificial jumps
    thetas_unwrapped = np.unwrap(thetas)
    # shift so that range is minimal (handle wrap-around)
    # try shifts of +2k*pi to align min/max span minimal
    # simpler: compute span via circular range:
    ang_sorted = np.sort(thetas)
    gaps = np.diff(np.concatenate([ang_sorted, ang_sorted[:1] + 2*np.pi]))
    # largest gap corresponds to region not covered; minimal covering arc is 2*pi - largest_gap
    largest_gap = np.max(gaps)
    min_arc_length = 2*np.pi - largest_gap
    # find endpoints of minimal arc: gap index where largest_gap occurs
    idx = np.argmax(gaps)
    # endpoints are ang_sorted[idx+1 mod m] .. ang_sorted[idx]
    m = len(ang_sorted)
    a_start = ang_sorted[(idx+1) % m]
    a_end = (ang_sorted[idx] if idx < m-1 else ang_sorted[idx] - 2*np.pi)
    # normalize so that a_end > a_start
    if a_end < a_start:
        a_end += 2*np.pi
    # minimal arc length is a_end - a_start (should be equal min_arc_length)
    arc_len = a_end - a_start
    # if arc_len > pi -> cannot cover with two rays (cone angle limited to pi)
    if arc_len > np.pi + 1e-9:
        return None, arc_len, (a_start, a_end)
    # pick representative points on curve closest to a_start and a_end
    # map all angles into same branch [a_start, a_start+2pi)
    thetas_shifted = np.mod(thetas - a_start + 2*np.pi, 2*np.pi)
    # we want points near 0 and near arc_len
    idx_start = np.argmin(np.abs(thetas_shifted - 0.0))
    idx_end   = np.argmin(np.abs(thetas_shifted - arc_len))
    ray1 = X[idx_start]
    ray2 = X[idx_end]
    return (ray1, ray2), arc_len, (idx_start, idx_end)

# Example
# if __name__ == "__main__":
#     A = np.array([[-1.0,  0.8],
#                   [ 0.0, -2.0]])
#     b = np.array([1.0, 0.6])
# 
#     A2 = np.array([
#         [0.0, 1.0],
#         [-0.1, -0.7]
#     ])
#     b2 = np.array([0, 1.5])
# 
#     t, X = sample_curve(A2, b2, T=40.0, num=3001)
#     res = two_rays_by_angle(X)
#     if res[0] is None:
#         print("Двух лучей недостаточно: угловой размах кривой = {:.3f} rad (> pi).".format(res[1]))
#     else:
#         (r1, r2), arc_len, (i1, i2) = res
#         print("Выбраны индексы точек:", i1, i2, "угловой размах (rad):", arc_len)
#         # визуализация
#         plt.figure(figsize=(6,6))
#         plt.plot(X[:,0], X[:,1], '-k', label='curve')
#         L = np.max(np.linalg.norm(X, axis=1)) * 1.05
#         for v, name in [(r1,'r1'), (r2,'r2')]:
#             plt.plot([0, v[0]], [0, v[1]], '--', lw=2, label=name)
#         # shade sector if arc small
#         theta1 = np.arctan2(r1[1], r1[0])
#         theta2 = np.arctan2(r2[1], r2[0])
#         # create sector polygon (fine grid)
#         ths = np.linspace(theta1, theta2, 60)
#         # handle wrap-around properly
#         if (theta2 - theta1) > np.pi:
#             # swap to take shorter arc
#             ths = np.linspace(theta2, theta1, 60)
#         poly = np.vstack([np.zeros(2), np.column_stack([L*np.cos(ths), L*np.sin(ths)])])
#         plt.fill(poly[:,0], poly[:,1], alpha=0.15)
#         plt.gca().set_aspect('equal')
#         plt.grid(True)
#         plt.legend()
#         plt.show()

# import numpy as np
# from scipy.linalg import expm, eig
# import matplotlib.pyplot as plt
# 
# def generate_curve(A, b, T=40.0, dt=0.01, tol=1e-6, max_steps=1_000_000):
#     """
#     Генерация кривой x(t) = exp(A t) b с досрочной остановкой,
#     если изменение точки становится меньше tol.
#     """
#     X = []
#     t_vals = []
#     prev = expm(A * 0.0) @ b
#     X.append(prev)
#     t_vals.append(0.0)
#     t = 0.0
#     steps = int(T / dt)
#     for i in range(1, min(steps, max_steps) + 1):
#         t = i * dt
#         cur = expm(A * t) @ b
#         if np.linalg.norm(cur - prev) < tol:
#             print(f"[INFO] Досрочная остановка на t={t:.2f}, изменение < {tol}")
#             break
#         X.append(cur)
#         t_vals.append(t)
#         prev = cur
#     return np.array(t_vals), np.vstack(X)
# 
# def two_rays_by_angle(X, eps=1e-12):
#     norms = np.linalg.norm(X, axis=1)
#     if np.any(norms < eps):
#         raise ValueError("Кривая проходит слишком близко к нулю — двух лучей может быть недостаточно.")
#     thetas = np.arctan2(X[:,1], X[:,0])
#     ang_sorted = np.sort(thetas)
#     gaps = np.diff(np.concatenate([ang_sorted, ang_sorted[:1] + 2*np.pi]))
#     largest_gap = np.max(gaps)
#     min_arc_length = 2*np.pi - largest_gap
#     idx = np.argmax(gaps)
#     m = len(ang_sorted)
#     a_start = ang_sorted[(idx+1) % m]
#     a_end = ang_sorted[idx] if idx < m-1 else ang_sorted[idx] - 2*np.pi
#     if a_end < a_start:
#         a_end += 2*np.pi
#     arc_len = a_end - a_start
#     if arc_len > np.pi + 1e-9:
#         return None, arc_len, (a_start, a_end)
#     thetas_shifted = np.mod(thetas - a_start + 2*np.pi, 2*np.pi)
#     idx_start = np.argmin(np.abs(thetas_shifted - 0.0))
#     idx_end   = np.argmin(np.abs(thetas_shifted - arc_len))
#     return (X[idx_start], X[idx_end]), arc_len, (idx_start, idx_end)
# 
# def plot_curve_with_rays(X, rays, A=None, show_eigvecs=False, title="Кривая и лучи"):
#     plt.figure(figsize=(6,6))
#     plt.plot(X[:,0], X[:,1], '-k', lw=1.5, label="Кривая exp(A t) b")
#     L = np.max(np.linalg.norm(X, axis=1)) * 1.05
#     for i, v in enumerate(rays):
#         plt.plot([0, v[0]], [0, v[1]], '--', lw=2, label=f"Луч {i+1}")
#     if show_eigvecs and A is not None:
#         w, v = eig(A)
#         for i in range(v.shape[1]):
#             vec = np.real(v[:,i])
#             if np.linalg.norm(vec) < 1e-12:
#                 continue
#             vec /= np.linalg.norm(vec)
#             plt.arrow(0, 0, L*vec[0], L*vec[1], 
#                       color='red', width=0.01, alpha=0.5 * (i + 1),
#                       label=f"Собственный вектор {i+1}")
#     plt.gca().set_aspect('equal')
#     plt.grid(True)
#     plt.legend()
#     plt.title(title)
#     plt.show()

import pandas as pd

def plot_traced_shuttle_points():
    def plot_trace(filename, label, color):
        df = pd.read_csv(filename, names=["step", "x", "y"])
        plt.plot(df["x"], df["y"], marker="o", linestyle="-", label=label, color=color, alpha=0.7)

    def plot_limits(limits_file="limits.csv"):
        try:
            df = pd.read_csv(limits_file, names=["x", "y"])
            plt.scatter(df["x"], df["y"], c="red", marker="*", s=120, label="Limits")
        except FileNotFoundError:
            pass

    # plt.figure(figsize=(7, 7))

    plot_trace("D:/Programming/C++/preisach_hysteresis_model/PreisachModel/trace_even.csv", "Even sequence", "blue")
    plot_trace("D:/Programming/C++/preisach_hysteresis_model/PreisachModel/trace_odd.csv", "Odd sequence", "green")
    plot_trace("D:/Programming/C++/preisach_hysteresis_model/PreisachModel/trace_all.csv", "All z_n", "gray")

    plot_limits()

    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.legend()
    plt.title("ShuttlePoint traces")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(True)
    # plt.show()

# --- Пример с вашей матрицей A2 ---
if __name__ == "__main__":
    cr = np.loadtxt('D:/Programming/C++/preisach_hysteresis_model/PreisachModel/out_curve.csv', delimiter=',', skiprows=1)
    rays = np.loadtxt('D:/Programming/C++/preisach_hysteresis_model/PreisachModel/out_rays.csv', delimiter=',', skiprows=1)

    t = cr[:, 0]
    X = cr[:, 1:3]

    plt.figure(figsize=(10, 10))
    plt.plot(X[:, 0], X[:, 1], '-k', label='curve')
    L = np.max(np.linalg.norm(rays, axis=1))
    for i in range(len(rays)):
        plt.plot([0, rays[i, 0] * L], [0, rays[i, 1] * L], '--', lw=2, label=f'ray{i}')
    plt.gca().set_aspect('equal', 'box')
    plt.grid(True)
    plt.legend()
    # plt.show()

    plot_traced_shuttle_points()
    plt.show()
    
    # A2 = np.array([
    #     [0.0, 1.0],
    #     [-0.1, -0.7]
    # ])
    # b2 = np.array([0, 1.5])
    # 
    # t, X = generate_curve(A2, b2, T=40.0, dt=0.05, tol=1e-5)
    # 
    # res = two_rays_by_angle(X)
    # if res[0] is None:
    #     print("Двух лучей недостаточно, угловой размах =", res[1])
    #     rays = []
    # else:
    #     rays, arc_len, idxs = res
    #     print("Угловой размах (рад):", arc_len)
    # 
    # if len(rays) > 0:
    #     plot_curve_with_rays(X, rays, A=A2, show_eigvecs=False, 
    #                          title="Кривая, лучи и собственные векторы")
