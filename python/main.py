import json
import os
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import ScalarFormatter
import scienceplots

from utils import TrajectoryAnalyzer, ConstrainedTrajectoryAnalyzer

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "Helvetica"
# })
# plt.style.use(['science', 'ieee', 'grid'])
# plt.rcParams.update({'figure.dpi': '500'})
# plt.rcParams.update({'font.size': 5})

import numpy as np
import polars as pl

from scipy.stats import linregress

from experiment_archive import plot_archive, read_result
from plotting import *
from utils import gen_steps_sequence, select_best_growth, calc_fourier_transform


def areal_model_test():
    with open("../PreisachModel/ArealModelTest.json") as f:
        data = json.load(f)
        # dt = data["dt"]
        # time = data["time"]
        # t = np.arange(0.0, time + dt, dt)
        t = data["times"]

        x = data['results']['x']
        # v = data['results']['v']

        inputs = data['anim']['in']
        derivatives = data['results']['derivatives']
        loop = data['loop']['outputs']

        # plt.plot(t, x)
        # plt.show()

        # plt.plot(t, v)
        # plt.show()
        # 
        # plt.plot(x, v)
        # plt.show()

        plt.plot(t, x)
        plt.plot(t, derivatives)
        plt.legend(["x", "derivatives"])
        plt.grid(True)
        plt.show()

        plt.plot(x, loop)
        plt.show()


def just_solve_rodos(path="../PreisachModel/JustSolveRodos.json"):
    if Path(path).suffix.lower() == ".zip":
        plot_archive(path)
        plt.show()
        return
    data = read_result(path)

    dt = data["dt"]
    time = data["time"]
    t = np.arange(0, time + dt, dt)

    x = np.asarray(data["results"]["x"])
    v = np.asarray(data["results"]["v"])
    # derivatives = data["results"]["derivatives"]
    inputs = data["loop"]["inputs"]
    loop = data["loop"]["outputs"]
    # ax = data["anim"]["x"]
    # ay = data["anim"]["y"]
    # aout = data["anim"]["out"]
    # plot_fourier_transform(x, dt=0.01, )

    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))

    path = "./article_plots"
    plot_x(x, t)
    plot_v(v, t)
    plot_phase_portrait(x, v)
    plot_hysteresis_loop(inputs, loop)
    plot_fourier_transform(x, dt=0.01)

    # everett_data = data["everett"]
    # x = np.asarray(everett_data["x"], dtype=np.float64)
    # y = np.asarray(everett_data["y"], dtype=np.float64)
    # n = len(x)
    # # JSON null -> Python None.
    # # Explicit conversion below restores null as np.nan.
    # z_flat = np.asarray(
    #     [
    #         np.nan if value is None else value
    #         for value in everett_data["z"]
    #     ],
    #     dtype=np.float64,
    # )
    # z = z_flat.reshape(n, n)
    #
    # X, Y = np.meshgrid(x, y)
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(
    #     111,
    #     projection="3d"
    # )
    #
    # z_min = np.nanmin(z)
    # z_max = np.nanmax(z)
    # norm = Normalize(
    #     vmin=z_min,
    #     vmax=z_max
    # )
    #
    #
    # surface = ax.plot_surface(
    #     X,
    #     Y,
    #     z,
    #     cmap="viridis",
    #     norm=norm,
    #     rstride=1,
    #     cstride=1,
    #     linewidth=0,
    #     antialiased=True
    # )
    #
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("E(x, y)")
    #
    # ax.set_title("Everett function")
    #
    # fig.colorbar(
    #     surface,
    #     ax=ax,
    #     shrink=0.65,
    #     pad=0.1,
    #     label="E(x, y)"
    # )
    #
    # plt.tight_layout()
    # plt.show()
    #
    # fig, ax = plt.subplots(figsize=(8, 7))
    #
    # contour = ax.contourf(
    #     X,
    #     Y,
    #     z,
    #     levels=40
    # )
    #
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_title("Everett function E(x, y)")
    #
    # ax.set_aspect("equal")
    #
    # fig.colorbar(
    #     contour,
    #     ax=ax,
    #     label="E(x, y)"
    # )
    #
    # plt.tight_layout()
    # plt.show()

    # scientific_plot(x, v, alpha=1, linewidth=.5, xlabel="x(t)", ylabel="v(t)")
    # scientific_plot(inputs, loop, alpha=1, linewidth=.5, xlabel="x(t)", ylabel="P[x(t)]")

    power_law = data["results"]["power_law"]
    Ts = np.asarray(power_law["Ts"])
    ns = np.asarray(power_law["ns"])
    slope = power_law["nu"]
    intercept = power_law["C"]

    plt.scatter(Ts, ns, marker='.')
    plt.plot(Ts, slope * Ts + intercept, 'r')
    plt.title(f"Slope ($\\nu$): {slope:.4f}, Intercept (C): {intercept:.4f}, $R^2$: {power_law["r2"]:.2f}.")
    plt.xlabel("$\\ln(T)$")
    plt.ylabel("$\\ln(\\delta x(t))$")
    plt.show()
    print()
        # 
        # trajs1 = power_law["trajs1"]
        # trajs2 = power_law["trajs2"]
        # 
        # timeKey = list(trajs1.keys())[-1]
        # 
        # traj1 = np.asarray(trajs1[timeKey])
        # traj2 = np.asarray(trajs2[timeKey])
        # 
        # x1, y1 = traj1[:, 0], traj1[:, 1]
        # x2, y2 = traj2[:, 0], traj2[:, 1]
        # 
        # n = 5000
        # x11 = np.asarray(x1[:n])
        # x22 = np.asarray(x2[:n])
        # y11 = np.asarray(y1[:n])
        # y22 = np.asarray(y2[:n])
        # 
        # plt.plot(x11, 'b')
        # plt.plot(x22, 'r')
        # plt.show()

        # LCEs = np.asarray(data["results"]["LCEs"])
        # LCEs_history = np.asarray(data["results"]["LCEs_history"])
        # 
        # plt.plot(LCEs_history[:, 0], c='red')
        # plt.plot(LCEs_history[:, 1], c='blue')
        # plt.xlabel("t")
        # plt.ylabel("$\\lambda$")
        # plt.annotate(f"{LCEs[0]:.5f}", xy=(len(LCEs_history[:, 0]), LCEs_history[-1, 0]), xytext=(400, -0.03), 
        #              arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"))
        # plt.annotate(f"{LCEs[1]:.5f}", xy=(len(LCEs_history[:, 1]), LCEs_history[-1, 1]), xytext=(400, -0.09), 
        #              arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2"))
        # plt.legend(["x", "$\\dot{x}$"])
        # plt.show()

        # k = 3000
        # plot_preisach_derivative(t[:(k+1)], inputs[:k], derivatives[:k])
        # sim = animate_preisach_plane(inputs, ax, ay, aout, 1.)
        # import matplotlib.animation as animation
        # writer = animation.FFMpegWriter(fps=30, metadata=dict(), bitrate=1800)
        # print("Saving hysteresis animation...")
        # sim.save(filename='preisach_sim.mp4', writer=writer)
        # print("Animation saved!")


def plot_lce():
    with open(f"../PreisachModel/RodosLCEs.json") as f:
        data = json.load(f)
        # dt = data["dt"]
        # E = data["E"]
        # time = data["time"]
        mLCE = data["mLCE"]
        # b_mLCE = data["results"]["Benettin_mLCE"]
        hist = data["hist"]
        # b_hist = data["results"]["Benettin_mLCE_history"]

        # t = np.arange(0.0, time, dt)

        print(f"mLCE = {mLCE}")
        plt.plot(hist, 'g')
        plt.plot(np.full_like(hist, mLCE), 'r--')
        plt.legend(["mLCE history", "mLCE"])
        plt.grid(True)
        plt.show()
        # print(f"Benettin mLCE = {b_mLCE}")
        # plt.plot(b_hist, 'b')
        # plt.show()

    # with open(f"../PreisachModel/RodosLCEs_TwoTrajs.json") as f:
    # with open(f"../PreisachModel/RodosLCEs_Regression.json") as f:
    #     data = json.load(f)
    #     dt = data["dt"]
    #     # time = data["time"]
    #     traj1 = np.asarray(data["traj1"])
    #     traj2 = np.asarray(data["traj2"])
    #     # t = np.arange(0.0, time, dt)
    # 
    #     x1, y1 = [i[0] for i in traj1], [i[1] for i in traj1]
    #     x2, y2 = [i[0] for i in traj2], [i[1] for i in traj2]
    # 
    #     x11 = np.asarray(x1[:4000])
    #     x22 = np.asarray(x2[:4000])
    #     y11 = np.asarray(y1[:4000])
    #     y22 = np.asarray(y2[:4000])
    # 
    #     k = int(0.5 / dt)
    #     # for i in range(0, k * 10, k):
    #     #     plt.vlines(i, -2, 1, colors='k', linestyles='dashed')
    #     # x1i = x1[i]
    #     # x2i = x2[i]
    #     # plt.plot(x1i, x2i, 'k')
    # 
    #     plt.plot(x11, y11, 'b')
    #     plt.plot(x22, y22, 'r')
    #     plt.show()
    # 
    #     plt.plot(np.sqrt(np.square(x11 - x22) + np.square(y11 - y22)))
    #     plt.show()


def n_regression(path="../PreisachModel/RodosLCEs_Regression.json"):
    data = read_result(path)

    Ts = np.asarray(data["Ts"])
    ns = np.asarray(data["ns"])
    M = data["M"]
    e = data["e"]

    # ------------regression--------------
    plt.scatter(Ts, ns)
    slope, intercept, r_value, _, _ = linregress(Ts, ns)

    print(ns / Ts)

    # model = np.polyfit(Ts, ns, 1)
    # print(model)
    #
    # predict = np.poly1d(model)
    # y_pred = predict(Ts)
    #
    # ss_total = np.sum((ns - np.mean(ns)) ** 2)
    # ss_res = np.sum((ns - y_pred) ** 2)
    # r_square = 1 - (ss_res / ss_total)
    # print(f"r_square = {r_square}")
    print(f"r_square = {r_value ** 2}")

    # plt.plot(Ts, y_pred, 'r')
    # plt.title(f"Slope (k): {model[0]:.4f}, Intercept (b): {model[1]:.4f}.\n M = {M}, e = {e:.2f}")
    # plt.show()
    plt.plot(Ts, slope * Ts + intercept, 'r')
    plt.title(f"Slope (k): {slope:.4f}, Intercept (b): {intercept:.4f}.\n M = {M}, e = {e:.2f}")
    plt.show()
    # ------------------------------------

    # ------------other--------------
    trajs1 = data["trajs1"]
    trajs2 = data["trajs2"]

    print(trajs1.keys())

    timeKey = list(trajs1.keys())[-1]

    # traj1 = np.asarray(trajs1["3.00"])
    # traj2 = np.asarray(trajs2["3.00"])
    traj1 = np.asarray(trajs1[timeKey])
    traj2 = np.asarray(trajs2[timeKey])
    # traj1 = np.asarray(trajs1["0.08"])
    # traj2 = np.asarray(trajs2["0.08"])
    x1, y1 = [i[0] for i in traj1], [i[1] for i in traj1]
    x2, y2 = [i[0] for i in traj2], [i[1] for i in traj2]

    n = -1
    x11 = np.asarray(x1[:n])
    x22 = np.asarray(x2[:n])
    y11 = np.asarray(y1[:n])
    y22 = np.asarray(y2[:n])

    # x11 = np.asarray(x1[:300])
    # x22 = np.asarray(x2[:300])
    # y11 = np.asarray(y1[:300])
    # y22 = np.asarray(y2[:300])

    plt.plot(x11, y11, 'b')
    plt.plot(x22, y22, 'r')
    plt.show()

    plt.plot(np.sqrt(np.square(x11 - x22) + np.square(y11 - y22)))
    plt.show()

    plt.plot(x11, 'b')
    plt.plot(x22, 'r')
    plt.show()
        # -------------------------------


def regressionTable():
    with open(f"../PreisachModel/RodosLCEs_DivergenceDegreeTable.json") as f:
        data = json.load(f)

        plot_areaCoeffsLoop(data)
        plot_epsilonLoop(data)


def two_trajs(path="../PreisachModel/TwoTrajs.json"):
    data = read_result(path)
    traj1 = np.asarray(data['traj1'])
    traj1_next = np.asarray(data['traj1_next'])
    traj2 = np.asarray(data['traj2'])

    x1 = traj1[:, 0][:10000]
    x1_next = traj1_next[:, 0][:10000]
    x2 = traj2[:, 0][:10000]

    # plt.plot(x1_next)
    # plt.plot(x2)
    # plt.show()

    # fig, ax = plt.subplots(1, 2, figsize=(8,8))
    # ax[0].plot(x1, 'r')
    # ax[0].plot(x1_next, 'b')
    #
    # ax[1].plot(x1, 'r')
    # ax[1].plot(x2, 'b')
    #
    # ax[0].grid(True)
    # ax[1].grid(True)
    # fig.show()

    plt.plot(x1 - x2)
    plt.show()

    print()


def TwoTrajsOnPhaseTrajectory(path="../PreisachModel/TwoTrajsOnPhaseTrajectory.json"):
    data = read_result(path)

    time = data["sample_times"] if "sample_times" in data else np.arange(0., float(data["time"]) + 0.001, 0.001)
    # x0s = np.asarray(data["x0s"])
    coords = np.asarray(data["coords"])
    # x0s = list(map(lambda x: f"[{x[0]:.3f}, {x[1]:.3f}]", x0s))
    # times = list(map(lambda t: f"{t:.3f}", list(data['times'])))
    trajs = dict()
    for (i, trs) in dict(data["trajs"]).items():
        trs = list(trs)
        trajs[int(i)] = [np.asarray(trs[0]), np.asarray(trs[1])]

    fig1, ax1 = plt.subplots(2, 5, figsize=(20, 20))
    fig2, ax2 = plt.subplots(2, 5, figsize=(20, 20))

    N = 0
    K = 10000

    dist_list = []
    x1_list = []
    x2_list = []
    for i in range(len(trajs)):
        traj1 = trajs[i][0]
        traj2 = trajs[i][1]

        x1 = traj1[:, 0][N:K]
        x2 = traj2[:, 0][N:K]
        v1 = traj1[:, 1][N:K]
        v2 = traj2[:, 1][N:K]

        dist = np.sqrt(np.square(x1 - x2) + np.square(v1 - v2))
        dist_list.append(dist)
        x1_list.append(x1)
        x2_list.append(x2)

    X_MIN, X_MAX = -1.0, 1.0

    ta = ConstrainedTrajectoryAnalyzer(time[N:K], dist_list, x1_list, x2_list)
    best_idx1 = ta.find_fastest_growth_integral(x_min=X_MIN, x_max=X_MAX)
    best_idx2 = ta.find_fastest_growth_weighted(alpha=0.5, x_min=X_MIN, x_max=X_MAX)

    for i in range(len(trajs)):
        j = i // 5
        k = i % 5

        # x0 = x0s[i]
        coord = coords[i]
        traj1 = trajs[i][0]
        traj2 = trajs[i][1]

        x1 = x1_list[i]
        x2 = x2_list[i]
        t = time[N:K]

        ax1[j, k].plot(t, x1, 'r')
        ax1[j, k].plot(t, x2, 'b')
        ax1[j, k].grid(True)
        ax1[j, k].set(xlabel="t", ylabel="x(t)")
        ax1[j, k].set_title(f"v = {coord}")

        dist = dist_list[i]
        if i == best_idx1:
            ax2[j, k].plot(t, dist, 'b')
        elif i == best_idx2:
            ax2[j, k].plot(t, dist, 'r')
        elif i == best_idx1 == best_idx2:
            ax2[j, k].plot(t, dist, 'g')
        else:
            ax2[j, k].plot(t, dist)

        ax2[j, k].grid(True)
        ax2[j, k].set(xlabel="t", ylabel="$x_1(t) - x_2(t)$")
        ax2[j, k].set_title(f"v = {coord}")

    fig1.show()
    fig2.show()

    x1 = x1_list[4]
    x2 = x2_list[4]
    t = time[N:K]
    plt.plot(t, x1, 'r')
    plt.plot(t, x2, 'b')
    plt.legend(["main", "perturbed"])
    plt.xlabel("t"),
    plt.ylabel("x(t)")
    plt.show()

    print()


def test_trajs_circle():
    with open(f"../PreisachModel/TestTrajsCircle.json") as f:
        data = json.load(f)

        es = np.asarray(data["es"])
        coords = np.asarray(data["coords"])

        series_list = []

        k = 2
        n = len(coords) // k
        for l, e in enumerate(es):
            trajs = data["trajs"]
            loops = data["loops"]
            div_fig, div_ax = plt.subplots(nrows=k, ncols=n, figsize=(35, 15))
            # phase_fig, phase_ax = plt.subplots(nrows=k, ncols=n, figsize=(35, 15))
            # x_fig, x_ax = plt.subplots(nrows=k, ncols=n, figsize=(35, 15))
            # loop_fig, loop_ax = plt.subplots(nrows=k, ncols=n, figsize=(35, 15))
            for i in range(len(coords)):
                m = i // n
                j = i % n

                v = coords[i]
                traj = trajs[v]

                loop = loops[v]
                in1, out1 = np.asarray(loop[0]), np.asarray(loop[1])
                in2, out2 = np.asarray(loop[2]), np.asarray(loop[3])

                main = np.asarray(traj[0])
                x1, y1 = np.asarray([i[0] for i in main]), np.asarray([i[1] for i in main])
                other = np.asarray(traj[1])
                x2, y2 = np.asarray([i[0] for i in other]), np.asarray([i[1] for i in other])

                div = np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))
                series_list.append(div)

                div_ax[m, j].plot(div)
                div_ax[m, j].set_title(f"v={v}")

                # phase_ax[m, j].plot(x1, y1, 'b')
                # phase_ax[m, j].plot(x2, y2, 'r')

                # x_ax[m, j].plot(x1, 'b')
                # x_ax[m, j].plot(x2, 'r')

                # loop_ax[m, j].plot(in1, out1, 'b')
                # loop_ax[m, j].plot(in2, out2, 'r')

            div_fig.suptitle(f"e={e}")
            # div_fig.legend()
            plt.show()
            # plt.savefig(f"./test_trajs_circle/test_trajs_circle_{l}.png")

        best_idx, best_segment, best_start_val, best_end_val, best_growth = select_best_growth(series_list)
        if best_idx is not None:
            print("Выбран временной ряд с индексом:", best_idx)
            print(
                f"Отрезок роста (start, end, slope, start_val, end_val, growth): ({best_segment}, {best_start_val}, {best_end_val}, {best_growth})")
        else:
            print("Подходящих временных рядов не найдено.")


def TwoTrajsOnCircle(path="../PreisachModel/TwoTrajsOnCircle.json"):
    data = read_result(path)

    coords = np.asarray(data["coords"])

    series_list = []

    k = 2
    n = len(coords) // k
    trajs = data["trajs"]
    loops = data["loops"]
    div_fig, div_ax = plt.subplots(nrows=k, ncols=n, figsize=(35, 15))
    # phase_fig, phase_ax = plt.subplots(nrows=k, ncols=n, figsize=(35, 15))
    x_fig, x_ax = plt.subplots(nrows=k, ncols=n, figsize=(35, 15))
    loop_fig, loop_ax = plt.subplots(nrows=k, ncols=n, figsize=(35, 15))
    for i in range(len(coords)):
        m = i // n
        j = i % n

        v = coords[i]
        traj = trajs[v]

        loop = loops[v]
        in1, out1 = np.asarray(loop[0]), np.asarray(loop[1])
        in2, out2 = np.asarray(loop[2]), np.asarray(loop[3])

        main = np.asarray(traj[0])
        x1, y1 = np.asarray([i[0] for i in main]), np.asarray([i[1] for i in main])
        other = np.asarray(traj[1])
        x2, y2 = np.asarray([i[0] for i in other]), np.asarray([i[1] for i in other])

        div = np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))[:700] * 100
        series_list.append(div)

        div_ax[m, j].plot(div)
        div_ax[m, j].set_title(f"v={v}")

        # phase_ax[m, j].plot(x1, y1, 'b')
        # phase_ax[m, j].plot(x2, y2, 'r')

        x_ax[m, j].plot(x1, 'b')
        x_ax[m, j].plot(x2, 'r')

        loop_ax[m, j].plot(in1, out1, 'b')
        loop_ax[m, j].plot(in2, out2, 'r')

    # div_fig.legend()
    plt.show()
    # plt.savefig(f"./test_trajs_circle/test_trajs_circle_{l}.png")

    best_idx, best_segment, best_start_val, best_end_val, best_growth = select_best_growth(series_list)
    if best_idx is not None:
        print("Выбран временной ряд с индексом:", best_idx)
        print(
            f"Отрезок роста (start, end, slope, start_val, end_val, growth): ({best_segment}, {best_start_val}, {best_end_val}, {best_growth})")
    else:
        print("Подходящих временных рядов не найдено.")


def PoincareMapping():
    with open(f"../PreisachModel/PoincareMapping.json") as f:
        data = json.load(f)
        pm = data["RadonsPoincareMapping"]
        x, v = np.asarray([i[0] for i in pm]), np.asarray([i[1] for i in pm])
        pm = np.vstack((x, v)).T

        min_x, min_y = pm.min(axis=0)
        max_x, max_y = pm.max(axis=0)
        range_x = max_x - min_x
        range_y = max_y - min_y
        pm_norm = (pm - [min_x, min_y]) / [range_x, range_y]

        x_norm, v_norm = np.asarray([i[0] for i in pm_norm]), np.asarray([i[1] for i in pm_norm])

        from scipy.spatial import KDTree
        tree = KDTree(pm_norm)
        dist, _ = tree.query(pm_norm, k=2)
        nearest_dist = dist[:, 1]

        r_obs = nearest_dist.mean()
        area = 1
        lam = len(pm_norm) / area
        r_exp = 1 / (2 * np.sqrt(lam))
        R = r_obs / r_exp
        print(R)

        # plt.scatter(x, v)
        # plt.show()

        plt.scatter(x_norm, v_norm)
        plt.show()


def BifurcationDiagram(path="../PreisachModel/BifurcationDiagram.json"):
    data = read_result(path)

    # param_name = data["param_name"]
    max_params = data["max_params"]
    min_params = data["min_params"]

    max_values = data["max_values"]
    min_values = data["min_values"]

    parameter = next((key for key in ("A", "E", "w", "rho", "d") if key in data), "A")
    plot_bifurcation_extrema(max_params, max_values, min_params, min_values, plt.gca(), x_label=parameter)
        
        # trajs = data["trajs"]
        # params = np.asarray(data["A"])
        # n = len(trajs)
        # m = len(trajs[0])
        # 
        # matrix = np.zeros((n, m))
        # for i in range(n):
        #     x = np.asarray(trajs[i])[:, 2]
        #     # if 25 <= params[i] <= 30:
        #     #     plt.plot(x[:10000])
        #     #     plt.show()
        #     matrix[i, :] = x
        # 
        # plot_bifurcation_diagram(params, matrix)

        # bifurcation_param = []
        # bifurcation_x = []
        # 
        # bifurcation_param_max = []
        # bifurcation_param_min = []
        # bif_max = []
        # bif_min = []
        # 
        # for i in range(len(params)):
        #     p = params[i]
        #     research = np.asarray(trajs[i])
        #     x = np.asarray([point[0] for point in research])
        # 
        #     maxima = argrelextrema(x, np.greater)
        #     minima = argrelextrema(x, np.less)
        #     extremas_idx = np.concatenate([maxima, minima], axis=None)
        #     x_peaks = x[extremas_idx]
        #     
        #     x_max = x[maxima]
        #     x_min = x[minima]
        #     bifurcation_param_max.extend([p] * len(x_max))
        #     bifurcation_param_min.extend([p] * len(x_min))
        #     bif_max.extend(x_max)
        #     bif_min.extend(x_min)
        #     
        # 
        #     bifurcation_param.extend([p] * len(x_peaks))
        #     bifurcation_x.extend(x_peaks)
        # 
        # # plt.figure(figsize=(10, 6))
        # plt.scatter(bifurcation_param_max, bif_max,
        #             s=0.1, c='red', linewidths=0, rasterized=True)
        # plt.scatter(bifurcation_param_min, bif_min,
        #             s=0.1, c='blue', linewidths=0, rasterized=True)
        # # plt.xlabel('$\\alpha$')
        # plt.xlabel('$\\omega$')
        # plt.ylabel("$x_{min}$, $x_{max}$")
        # # plt.title('Bifurcation Diagram')
        # # plt.grid(True)
        # # plt.tight_layout()
        # # plt.show()
        # 
        # path = "./article_plots"
        # plt.savefig(f"{path}/biff_w.png")


def ShiftTest():
    with open(f"../PreisachModel/ShiftTest.json") as f:
        data = json.load(f)
        z_minus = data["z-"]
        z_plus = data["z+"]

        x1 = np.array(z_minus["x"])
        v1 = np.array(z_minus["v"])

        print(x1[-1], v1[-1])

        x2 = np.array(z_plus["x"])
        v2 = np.array(z_plus["v"])
        # print(x2[-1], v2[-1])

        x0 = "{0.8, 0.8}"
        T = "T"
        gamma = 4.
        omega0 = 3.

        plt.figure(figsize=(12, 12))
        plt.suptitle(f"x0 = -{x0}, T = {T}\n$\\gamma = {gamma}, \\omega_0 = {omega0}, \\omega = \\omega_0, A = 1.5$")

        plt.subplot(121)
        plt.plot(x1, v1)
        plt.plot(-1.190174627449658, 0.13896378639632018, 'x', color='black', label='End')
        plt.plot(x1[0], v1[0], 'o', color='green', label='Begin')
        plt.plot(x1[-1], v1[-1], 'x', color='red', label='End')
        plt.xlabel('x')
        plt.ylabel('v')
        plt.grid()
        plt.title(f"z-")

        plt.subplot(122)
        plt.plot(x2, v2)
        plt.plot(x2[0], v2[0], 'o', color='green', label='Begin')
        plt.plot(x2[-1], v2[-1], 'x', color='red', label='End')
        plt.xlabel('x')
        plt.ylabel('v')
        plt.grid()
        plt.title(f"z+")

        plt.show()


def ShuttlePointLog():
    from matplotlib.lines import Line2D

    _SEQ_RE = re.compile(r"y_\{(\d+),(\d+)\}")
    annotate_aux = True

    def parse_seq_name(name):
        m = _SEQ_RE.fullmatch(name)
        if m is None:
            raise ValueError(f"Unknown sequence name: {name}")

        return int(m.group(1)), int(m.group(2))

    def as_trajectory(a):
        """
        Возвращает траекторию в форме (N, 2).

        В логах ShiftTraj и Forward у вас, судя по коду,
        могут иметь разную ориентацию матриц.
        """
        a = np.asarray(a)

        if a.ndim != 2:
            raise ValueError(f"Trajectory has shape {a.shape}")

        if a.shape[1] == 2:
            return a

        if a.shape[0] == 2:
            return a.T

        raise ValueError(f"Cannot interpret trajectory shape {a.shape}")

    with open(f"../PreisachModel/ShuttlePointLog.json") as f:
        data = json.load(f)

        shuttle = data['shuttle_logs']

        cone = shuttle["cone"]
        ray1 = np.asarray(cone["rays"][0], dtype=float)
        ray2 = np.asarray(cone["rays"][1], dtype=float)

        traj_z_minus = as_trajectory(shuttle["traj_z-"])
        traj_z_plus = as_trajectory(shuttle["traj_z+"])
        
        limits = shuttle["limits"]
        lim_even = np.asarray(limits[0], dtype=float)
        lim_odd = np.asarray(limits[1], dtype=float)

        sequences = {
            k: as_trajectory(v)
            for k, v in shuttle["sequences"].items()
        }

        initial_points = {
            k: np.asarray(v, dtype=float)
            for k, v in shuttle["initial_points"].items()
        }

        cond_points = {
            k: np.asarray(v, dtype=float)
            for k, v in shuttle["cond_points"].items()
        }

        zs = [
            np.asarray(v, dtype=float)
            for v in shuttle["zs"]
        ]

        # ----------------------------------------------------------
        # Сортируем вспомогательные последовательности
        # по номеру z_n и внутренней итерации i.
        # ----------------------------------------------------------
        sequence_names = sorted(
            sequences.keys(),
            key=parse_seq_name,
        )

        fig, ax = plt.subplots(figsize=(13, 10))

        ax.axhline(0.0, linewidth=0.8, alpha=0.25)
        ax.axvline(0.0, linewidth=0.8, alpha=0.25)

        # ==========================================================
        # Сначала собираем диапазон, чтобы красиво продлить конус.
        # ==========================================================
        all_points = []

        for trajectory in sequences.values():
            all_points.append(trajectory)

        all_points.append(traj_z_minus)
        all_points.append(traj_z_plus)

        all_points.append(lim_odd)
        all_points.append(lim_even)

        if zs:
            all_points.append(np.asarray(zs))

        xy = np.vstack(all_points)

        xmin, ymin = xy.min(axis=0)
        xmax, ymax = xy.max(axis=0)

        span = max(xmax - xmin, ymax - ymin)

        if span == 0:
            span = 1.0

        # ==========================================================
        # Конус
        # ==========================================================
        def normalized(v):
            n = np.linalg.norm(v)
            if n == 0:
                return v
            return v / n

        cone_length = 0.55 * span

        r1 = normalized(ray1) * cone_length
        r2 = normalized(ray2) * cone_length

        ax.plot([0, r1[0]], [0, r1[1]], linestyle="--", linewidth=1.4, alpha=0.7)

        ax.plot([0, r2[0]], [0, r2[1]], linestyle="--", linewidth=1.4, alpha=0.7)

        ax.fill([0, r1[0], r2[0]], [0, r1[1], r2[1]], alpha=0.06, label=r"$K$")

        # ==========================================================
        # Исходные z- и z+
        # ==========================================================
        z_minus = traj_z_minus[0]
        z_plus = traj_z_plus[0]

        ax.scatter(*z_minus, marker="s", s=75, zorder=7)
        ax.annotate(r"$z_-$", z_minus, xytext=(-18, 12), textcoords="offset points", fontsize=13, fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.2",
                fc="white",
                alpha=0.85,
                ec="none",
            ),
        )

        ax.scatter(*z_plus, marker="s", s=75, zorder=7)
        ax.annotate(r"$z_+$", z_plus, xytext=(10, 12), textcoords="offset points", fontsize=13, fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.2",
                fc="white",
                alpha=0.85,
                ec="none",
            ),
        )

        # ==========================================================
        # Вспомогательные последовательности y_i^(n)
        # ==========================================================
        odd_label_used = False
        even_label_used = False

        for name in sequence_names:
            v_index, inner_index = parse_seq_name(name)

            trajectory = sequences[name]
            p = initial_points[name]

            is_odd = (v_index % 2 == 1)

            if is_odd:
                line_color = "tab:orange"
                point_color = "tab:red"

                line_label = (
                    r"вспомогательная последовательность для $z_{2k+1}$"
                    if not odd_label_used
                    else None
                )

                odd_label_used = True
            else:
                line_color = "tab:blue"
                point_color = "tab:cyan"

                line_label = (
                    r"вспомогательная последовательность для $z_{2k}$"
                    if not even_label_used
                    else None
                )

                even_label_used = True

            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                color=line_color,
                linewidth=1.0,
                alpha=0.22,
                zorder=1,
                label=line_label,
            )

            ax.scatter(
                p[0],
                p[1],
                s=22,
                color=point_color,
                alpha=0.65,
                zorder=3,
            )

            # Подписываем не каждый элемент, иначе график
            # быстро становится нечитаемым.
            # первые два + затем каждый пятый
            should_annotate = (
                inner_index <= 2
                or inner_index % 5 == 0
            ) if annotate_aux else True

            if should_annotate:
                # Смещаем подпись в разные стороны,
                # чтобы соседние подписи не лежали друг на друге.
                sign = 1 if inner_index % 2 else -1

                dx = 7 + 3 * (inner_index % 3)
                dy = sign * (7 + 2 * (v_index % 3))

                ax.annotate(
                    rf"$y_{{{v_index}, {inner_index}}}$",
                    p,
                    xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=8,
                    alpha=0.7,
                    bbox=dict(
                        boxstyle="round,pad=0.15",
                        fc="white",
                        ec="none",
                        alpha=0.65,
                    ),
                )

        # ==========================================================
        # Челночная последовательность z_n
        # ==========================================================

        for j, z in enumerate(zs):
            if j == 0:
                marker = "D"
                size = 65
            elif j % 2 == 0:
                marker = "o"
                size = 65
            else:
                marker = "^"
                size = 70

            ax.scatter(
                z[0],
                z[1],
                marker=marker,
                s=size,
                edgecolor="black",
                linewidth=0.7,
                zorder=10,
            )

            # Разносим подписи чётной и нечётной ветвей
            if j == 0:
                offset = (-15, -22)
            elif j % 2 == 0:
                offset = (-25, -20)
            else:
                offset = (10, 12)

            ax.annotate(
                rf"$z_{{{j}}}$",
                z,
                xytext=offset,
                textcoords="offset points",
                fontsize=11,
                fontweight="bold",
                zorder=11,
                arrowprops=dict(
                    arrowstyle="-",
                    linewidth=0.6,
                    alpha=0.6,
                ),
                bbox=dict(
                    boxstyle="round,pad=0.22",
                    fc="white",
                    ec="0.75",
                    alpha=0.5,
                ),
            )

        # Соединяем только элементы одной подпоследовательности:
        # z0 -> z2 -> z4 -> ...
        # z1 -> z3 -> z5 -> ...

        even_zs = np.asarray(zs[0::2])
        if len(even_zs) > 1:
            ax.plot(
                even_zs[:, 0],
                even_zs[:, 1],
                linestyle="-",
                linewidth=2.0,
                alpha=0.75,
                label=r"$z_{2k}$",
                zorder=6,
            )

        odd_zs = np.asarray(zs[1::2])
        if len(odd_zs) > 1:
            ax.plot(
                odd_zs[:, 0],
                odd_zs[:, 1],
                linestyle="-",
                linewidth=2.0,
                alpha=0.75,
                label=r"$z_{2k+1}$",
                zorder=6,
            )

        # ==========================================================
        # Предельные точки
        # ==========================================================
        ax.scatter(*lim_odd, color='red', marker="*", s=75, zorder=20)
        ax.annotate(r"$L_{-}$", lim_odd, color='red', zorder=20, xytext=(-18, 8), textcoords="offset points", fontsize=11, fontweight="bold")

        ax.scatter(*lim_even, color='red', marker="*", s=75, zorder=20)
        ax.annotate(r"$L_{+}$", lim_even, color='red', zorder=20, xytext=(10, 8), textcoords="offset points", fontsize=11, fontweight="bold")

        # ==========================================================
        # Оформление
        # ==========================================================

        margin = 0.08 * span

        ax.set_xlim(xmin - margin, xmax + margin)
        ax.set_ylim(ymin - margin, ymax + margin)

        ax.set_xlabel(r"$x$", fontsize=13)
        ax.set_ylabel(r"$\dot{x}$", fontsize=13)

        ax.set_title(
            "Челночный алгоритм: вспомогательные и челночные последовательности",
            fontsize=15,
            pad=14,
        )

        ax.grid(
            True,
            linewidth=0.6,
            alpha=0.25,
        )

        # Для фазового портрета часто удобнее одинаковый масштаб.
        # Если x и dx/dt физически сильно различаются по масштабу,
        # эту строку можно убрать.
        ax.set_aspect("equal", adjustable="datalim")

        ax.legend(
            loc="best",
            framealpha=0.92,
            fontsize=10,
        )

        fig.tight_layout()

        plt.show()


def article_plots():
    json_path = "../PreisachModel/JustSolveRodos.json"
    with open(json_path) as f:
        data = json.load(f)

        prefix = "double_loop_1"
        num = "ninth"
        name = f"{prefix}_{num}_set"

        dt = data["dt"]
        time = data["time"]
        t = np.arange(0, time + dt, dt)

        x = np.asarray(data["results"]["x"])
        v = np.asarray(data["results"]["v"])
        inputs = data["loop"]["inputs"]
        loop = data["loop"]["outputs"]

        path = Path(f"./article_plots/{name}")
        if not path.exists():
            os.mkdir(path)

        shutil.copy(json_path, path)

        scientific_plot(t, x, alpha=1, linewidth=.5, xlabel="t", ylabel="x(t)", save=f"{path}/{name}_x.png")
        scientific_plot(t, v, alpha=1, linewidth=.5, xlabel="t", ylabel="v(t)", save=f"{path}/{name}_v.png")
        scientific_plot(x, v, alpha=1, linewidth=.5, xlabel="x(t)", ylabel="v(t)", save=f"{path}/{name}_phase.png")
        scientific_plot(inputs, loop, alpha=1, linewidth=1, xlabel="x(t)", ylabel="P[x(t)]",
                        save=f"{path}/{name}_loop.png")


def OperatorDerivativeTest(path="../PreisachModel/OperatorDerivativeTest.json"):
    data = read_result(path)
    x = data["results"]["x"]
    derivatives = data["results"]["derivatives"]
    inputs = data["loop"]["inputs"]
    loop = data["loop"]["outputs"]

    plt.plot(x)
    plt.plot(derivatives, 'r')
    plt.show()


def ZeroOneTest(path="../PreisachModel/ZeroOneTest.json"):
    data = read_result(path)

    # x = np.asarray(data["x"])[:50000]
    # v = np.asarray(data["v"])[:50000]
    #
    # plt.plot(x)
    # plt.show()
    # plt.plot(v)
    # plt.show()
    # plt.plot(x, v)
    # plt.show()

    Ks = data["Ks"]
    param = data["A"]

    plot_zero_one(param, Ks, plt.gca())


def plot_bifurcation_diagram(
    bifurcation_params: np.ndarray,
    time_series_matrix: np.ndarray,
    title: str = "Bifurcation Diagram",
    output_file: str = None
):
    """
    Строит бифуркационную диаграмму на основе локальных экстремумов.

    Args:
        bifurcation_params: 1D массив значений параметра (N,).
        time_series_matrix: 2D матрица (N, T), где строка i соответствует решению при params[i].
        title: Заголовок графика.
        output_file: Если указано, сохраняет график в файл.
    """

    # Проверка размерности
    if time_series_matrix.shape[0] != bifurcation_params.shape[0]:
        raise ValueError(
            f"Несоответствие размерностей: params {bifurcation_params.shape}, matrix {time_series_matrix.shape}")

    # Векторизованный поиск экстремумов
    # Сравниваем элементы [1:-1] с их левыми [:-2] и правыми [2:] соседями
    data_center = time_series_matrix[:, 1:-1]
    data_left = time_series_matrix[:, :-2]
    data_right = time_series_matrix[:, 2:]

    # Логические маски для строгих локальных максимумов и минимумов
    # Используем строгое неравенство для фильтрации плато (хотя численный шум может влиять)
    is_max = (data_center > data_left) & (data_center > data_right)
    is_min = (data_center < data_left) & (data_center < data_right)

    # Получаем индексы строк (соответствуют параметрам) и значения экстремумов
    rows_max, _ = np.where(is_max)
    vals_max = data_center[is_max]
    params_max = bifurcation_params[rows_max]

    rows_min, _ = np.where(is_min)
    vals_min = data_center[is_min]
    params_min = bifurcation_params[rows_min]

    # Визуализация
    plt.figure(figsize=(10, 6))

    # s=0.5 и alpha=0.5 позволяют видеть плотность распределения точек
    plt.scatter(params_max, vals_max, c='red', s=0.5, alpha=0.6, label='Local Maxima', marker='.')
    plt.scatter(params_min, vals_min, c='blue', s=0.5, alpha=0.6, label='Local Minima', marker='.')

    plt.title(title)
    plt.xlabel('Bifurcation Parameter')
    plt.ylabel('State Variable (Extrema)')
    plt.legend(markerscale=10)  # Увеличиваем маркеры в легенде для видимости
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file)
        print(f"График сохранен в {output_file}")
    else:
        plt.show()


def TestBiffDeidram():
    n_params = 1000
    n_steps = 200  # Длина временного ряда (после отсечения переходного процесса)
    transient = 100  # Переходный процесс

    r_values = np.linspace(2.5, 4.0, n_params)

    matrix = np.zeros((n_params, n_steps))

    # Начальное состояние для всех r
    x = 0.5 * np.ones(n_params)

    for _ in range(transient):
        x = r_values * x * (1 - x)

    for t in range(n_steps):
        x = r_values * x * (1 - x)
        matrix[:, t] = x

    print("Построение диаграммы...")
    plot_bifurcation_diagram(r_values, matrix)


def parse_shuttle_point_initial_values():
    # 1. Читаем файл как сырой текст для кастомного парсинга.
    # Предполагаем, что разделитель — ';', заголовков нет
    raw_df = pl.read_csv(
        "../PreisachModel/trace_zs.csv",
        has_header=False,
        separator=";",
        new_columns=["u0_raw", "z_raw", "flag1", "flag2"]
    )

    # 2. Очищаем данные и приводим к нужным типам
    # Извлекаем числа из строк вида "{0.000, 0.000}"
    df = raw_df.with_columns([
        # Парсим вектор u0 как список из двух Float64
        pl.col("u0_raw")
        .str.extract_all(r"[-+]?\d*\.\d+|\d+")
        .cast(pl.List(pl.Float64))
        .alias("u0"),

        # Парсим вектор z как список из двух Float64
        pl.col("z_raw")
        .str.extract_all(r"[-+]?\d*\.\d+|\d+")
        .cast(pl.List(pl.Float64))
        .alias("z"),

        # Приводим флаги к Boolean
        pl.col("flag1").cast(pl.Boolean),
        pl.col("flag2").cast(pl.Boolean)
    ]).drop(["u0_raw", "z_raw"])

    # Фильтруем (оставляем строки, где хотя бы один флаг True)
    df_filtered = df.filter(pl.col("flag1") | pl.col("flag2"))

    # Добавляем уникальный ID для каждой строки, чтобы корректно разделять их при Join
    df_filtered = df_filtered.with_row_index("row_id")

    # 2. Переводим u0 в Struct, так как по List делать Join в Polars нельзя
    df_filtered = df_filtered.with_columns(
        pl.col("u0").list.to_struct(fields=["u0_x", "u0_y"])
    )

    # 3. Делаем Self-Join по вектору u0 для поиска всех возможных пар
    pairs = df_filtered.join(
        df_filtered,
        on="u0",
        suffix="_right"
    )

    # 4. Фильтрация получившихся пар по вашему ТЗ
    filtered_pairs = pairs.filter(
        # Исключаем соединение строки самой с собой и зеркальные дубликаты (пара A-B и B-A)
        (pl.col("row_id") < pl.col("row_id_right")) &
        (
            # Условие 1: Левая строка — это (True, True)
            (pl.col("flag1") & pl.col("flag2")) |
            # Условие 2: Правая строка — это (True, True)
            (pl.col("flag1_right") & pl.col("flag2_right")) |
            # Условие 3: Флаги просто различаются
            ((pl.col("flag1") != pl.col("flag1_right")) | (pl.col("flag2") != pl.col("flag2_right")))
        )
    )

    # 5. Собираем финальный результат
    # Разворачиваем пары обратно в плоскую таблицу строк
    result = pl.concat([
        filtered_pairs.select(["u0", "z", "flag1", "flag2"]),
        filtered_pairs.select([
            pl.col("u0"),
            pl.col("z_right").alias("z"),
            pl.col("flag1_right").alias("flag1"),
            pl.col("flag2_right").alias("flag2")
        ])
    ]).unique()  # unique() уберет пересечения, если одна строка вошла в несколько разных пар

    result = result.sort("u0")

    # 6. Возвращаем u0 исходный тип List[Float64]
    result = result.with_columns(
        pl.concat_list([pl.col("u0").struct.field("u0_x"), pl.col("u0").struct.field("u0_y")]).alias("u0")
    )

    print(result)


if __name__ == '__main__':
    # TwoTrajsOnPhaseTrajectory()
    # two_trajs()
    # ComplexRadonsResearch()
    # ZeroOneTest()
    # TestBiffDeidram()

    # article_plots()
    # parse_shuttle_point_initial_values()
    # ShuttlePointLog()

    # BifurcationDiagram()
    # OperatorDerivativeTest()

    # DoublePreisachModelTest()

    # ShiftTest()

    just_solve_rodos()

    # eps = 0.5
    # TF = 50.0
    # with (open(f"./pl/PowerLaw_{eps}_{TF}.json") as f):
    #     data = json.load(f)
    #     data = data['results']
    #     powerLaw = {
    #         "e": float(data["LCEs"]["power_law"]["e"]),
    #         "M": int(data["LCEs"]["power_law"]["M"]),
    #         "Ts": np.asarray(data["LCEs"]["power_law"]["Ts"]),
    #         "ns": np.asarray(data["LCEs"]["power_law"]["ns"]),
    #         "r2": np.asarray(data["LCEs"]["power_law"]["r2"]),
    #         "nu": float(data["LCEs"]["power_law"]["nu"]),
    #         "C": float(data["LCEs"]["power_law"]["C"]),
    #         "trajs1": dict((name, np.asarray(traj)) for (name, traj) in dict(data["LCEs"]["power_law"]["trajs1"]).items()),
    #         "trajs2": dict((name, np.asarray(traj)) for (name, traj) in dict(data["LCEs"]["power_law"]["trajs2"]).items()),
    #     }
    # 
    #     fig, ax = plt.subplots(2,2, figsize=(8,8))
    #     ax[0, 0].scatter(powerLaw["Ts"], powerLaw["ns"])
    #     slope, intercept, r_value, _, _ = linregress(powerLaw["Ts"], powerLaw["ns"])
    #     ax[0, 0].plot(powerLaw["Ts"], slope * powerLaw["Ts"] + intercept, 'r')
    #     ax[0, 0].plot(powerLaw["Ts"], powerLaw['nu'] * powerLaw["Ts"] + powerLaw['C'], 'b')
    #     ax[0, 0].set(xlabel="T", ylabel="$\\nu$")
    #     ax[0, 0].set_title(f"$\\nu$ = {powerLaw['nu']},\n C = {powerLaw['C']},\n $R^2$ = {powerLaw['r2']}")
    #     ax[0, 0].grid(True)
    # 
    #     print(f"nu: {powerLaw['nu']}, C: {powerLaw['C']}, r2: {powerLaw['r2']}")
    # 
    #     for (i, T) in enumerate(['1.00', '4.00', '2.50']):
    #         j = (i + 1) // 2
    #         k = (j + i + 1) % 2
    # 
    #         traj1 = powerLaw['trajs1'][T][:, 0]
    #         traj2 = powerLaw['trajs2'][T][:, 0]
    #         ax[j, k].plot(traj1[:1000], 'r')
    #         ax[j, k].plot(traj2[:1000], 'b')
    #         ax[j, k].legend(['main traj', 'perturbed traj'])
    #         ax[j, k].set_title(f"T = {T}")
    #         ax[j, k].grid(True)
    # 
    #     fig.savefig(f"./pl/fig2_tf.png", dpi=300)
    #         
    # 
    #     fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    #     for (i, T) in enumerate(['1.00', '4.00', '2.50']):
    #         j = (i + 1) // 2
    #         k = (j + i + 1) % 2
    # 
    #         traj1 = powerLaw['trajs1'][T][:, 0]
    #         traj2 = powerLaw['trajs2'][T][:, 0]
    # 
    #         freq, fourier = calc_fourier_transform(traj1, dt=0.01) 
    #         ax[j, k].plot(freq[:300], fourier[:300], '-r')
    #         freq, fourier = calc_fourier_transform(traj2, dt=0.01)
    #         ax[j, k].plot(freq[:300], fourier[:300], '--b')
    #         ax[j, k].legend(['main traj', 'perturbed traj'])
    # 
    #         ax[j, k].set_title(f"T = {T}")
    #         ax[j, k].grid(True)
    # 
    #     fig.suptitle(f"$\\varepsilon$ = {eps}")
    #     fig.savefig(f"./pl/fig_f2_tf.png", dpi=300)
