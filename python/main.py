import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science', 'ieee' ,'grid'])
# plt.rcParams.update({'figure.dpi': '100'})
# plt.rcParams.update({'font.size': 18})

import numpy as np
import glob

from scipy.signal import argrelextrema
from scipy.stats import linregress

from plotting import *
from utils import gen_steps_sequence, select_best_growth


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


def just_solve_rodos():
    with open(f"../PreisachModel/JustSolveRodos.json") as f:
        data = json.load(f)

        dt = data["dt"]
        time = data["time"]
        t = np.arange(0, time + dt, dt)

        # h = data["h"]
        E = data["E"]
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


def n_regression():
    with open(f"../PreisachModel/RodosLCEs_Regression.json") as f:
        data = json.load(f)

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


def two_trajs():
    with open(f"../PreisachModel/RodosLCEs_TwoTrajs.json") as f:
        data = json.load(f)

        e = float(data["e"])
        M = int(data["M"])
        T = float(data["T"])
        dt = float(data["dt"])
        timeForward = float(data["timeForward"]) if "timeForward" in data else 0
        time = int(M * T / dt)

        k = int(T / dt)

        traj1 = np.asarray([np.asarray(x) for x in data["traj1"]])
        traj2 = np.asarray([np.asarray(x) for x in data["traj2"]])

        x1, v1 = np.asarray([i[0] for i in traj1]), np.asarray([i[1] for i in traj1])
        x2, v2 = np.asarray([i[0] for i in traj2]), np.asarray([i[1] for i in traj2])

        x1, v1 = x1[:10000], v1[:10000]
        x2, v2 = x2[:(10000 - timeForward)], v2[:10000]
        time = np.arange(0, 10000, 1)

        plt.plot(time, x1, 'b')
        plt.plot(time[timeForward:], x2, 'r')
        plt.show()


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


def TwoTrajsOnCircle():
    with open(f"../PreisachModel/TwoTrajsOnCircle.json") as f:
        data = json.load(f)

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


def BifurcationDiagram():
    with open(f"../PreisachModel/BifurcationDiagram.json") as f:
        data = json.load(f)
        trajs = data["trajs"]
        params = np.asarray(data["rho"])
        n = len(trajs)
        m = len(trajs[0])
        
        matrix = np.zeros((n, m))
        for i in range(n):
            x = np.asarray(trajs[i])[:, 2]
            if 25 <= params[i] <= 30:
                plt.plot(x[:10000])
                plt.show()
            matrix[i, :] = x
        
        plot_bifurcation_diagram(params, matrix)

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
        

def ShuttlePointTraj():
    with open(f"../PreisachModel/ShuttlePointTraj.json") as f:
        data = json.load(f)
        x = np.array(data["x"])
        v = np.array(data["v"])
        
        loop_in = np.array(data["loop"]["in"])
        loop_out = np.array(data["loop"]["out"])

        plt.plot(x)
        plt.xlabel('t')
        plt.ylabel('x')
        plt.show()
        plt.plot(x, v)
        plt.xlabel('x')
        plt.ylabel('v')
        plt.show()
        plt.plot(loop_in, loop_out)
        plt.xlabel('in')
        plt.ylabel('out')
        plt.show()


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
    with open(f"../PreisachModel/ShuttlePointLog.json") as f:
        data = json.load(f)
        
        iters = data["iters"]
        
        cone = data["cone"]
        ray1 = np.asarray(cone["rays"][0])
        ray2 = np.asarray(cone["rays"][1])

        traj_z_minus = np.asarray(data["traj_z-"])
        traj_z_plus = np.asarray(data["traj_z+"])
        new_z_minus = np.asarray(data["new_z-"])
        new_z_plus = np.asarray(data["new_z+"])

        odd_sequences = []
        for seq in list(data["odd_sequences"]):
            odd_sequences.append(np.asarray(seq))

        odd_initial_points = []
        for seq in list(data["odd_initial_points"]):
            odd_initial_points.append(np.asarray(seq))

        odd_cond_points = []
        for seq in list(data["odd_cond_points"]):
            odd_cond_points.append(np.asarray(seq))

        even_sequences = []
        for seq in list(data["even_sequences"]):
            even_sequences.append(np.asarray(seq))

        even_initial_points = []
        for seq in list(data["even_initial_points"]):
            even_initial_points.append(np.asarray(seq))

        even_cond_points = []
        for seq in list(data["even_cond_points"]):
            even_cond_points.append(np.asarray(seq))
            
        
        # plt.plot(traj_z_plus[0], traj_z_plus[1], color='brown', alpha=0.3)
        # plt.show()
        # plt.plot(traj_z_plus[0], color='brown', alpha=0.3)
        # plt.plot(traj_z_plus[1], color='brown', alpha=0.3)
        # plt.show()


        plt.figure(figsize=(12, 12))
            
        plt.plot([0, ray1[0]], [0, ray1[1]], color='black', label='Ray1')
        plt.plot([0, ray2[0]], [0, ray2[1]], color='black', label='Ray2')
        
        plt.plot(traj_z_minus[0, 0], traj_z_minus[1, 0], 'o', color='green')
        plt.plot(traj_z_minus[0], traj_z_minus[1], color='brown', alpha=0.3)
        plt.plot(new_z_minus[0], new_z_minus[1], 'x', color='red')
        plt.annotate("z-", xy=(traj_z_minus[0, 0] - 0.05, traj_z_minus[1, 0] - 0.05))
        plt.annotate("new z-", xy=(new_z_minus[0] - 0.05, new_z_minus[1] - 0.05))

        plt.plot(traj_z_plus[0, 0], traj_z_plus[1, 0], 'o', color='green')
        plt.plot(traj_z_plus[0], traj_z_plus[1], color='brown', alpha=0.3)
        plt.plot(new_z_plus[0], new_z_plus[1], 'x', color='red')
        plt.annotate("z+", xy=(traj_z_plus[0, 0] + 0.05, traj_z_plus[1, 0] + 0.05))
        plt.annotate("new z+", xy=(new_z_plus[0] - 0.05, new_z_plus[1] - 0.05))
        
        for i in range(1, iters + 1):
            if i % 2 != 0 and i < len(odd_sequences):
                j = i // 2 - i % 2 + 1
                plt.plot(odd_sequences[j][0], odd_sequences[j][1], color='purple', label='Odd Sequence', alpha=0.3)
                plt.plot(odd_initial_points[j][0], odd_initial_points[j][1], 'o', color='coral',
                         label='Odd New Initial Point')
                plt.plot(odd_cond_points[j][0], odd_cond_points[j][1], 'o', color='lime',
                         label='Odd Cond Initial Point')
                plt.annotate(f"{i}", xy=(odd_initial_points[j][0], odd_initial_points[j][1]))
                plt.annotate(f"{i}", xy=(odd_cond_points[j][0], odd_cond_points[j][1]))
            elif i % 2 == 0 and i < len(even_sequences):
                j = i // 2 - i % 2 - 1
                plt.plot(even_sequences[j][0], even_sequences[j][1], color='purple', label='Even Sequence', alpha=0.3)
                plt.plot(even_initial_points[j][0], even_initial_points[j][1], 'o', color='crimson',
                         label='Even New Initial Point')
                plt.plot(even_cond_points[j][0], even_cond_points[j][1], 'o', color='yellowgreen',
                         label='Even Cond Initial Point')
                plt.annotate(f"{i}", xy=(even_initial_points[j][0], even_initial_points[j][1]))
                plt.annotate(f"{i}", xy=(even_cond_points[j][0], even_cond_points[j][1]))
        
        # for i in range(len(odd_sequences)):
        #     plt.plot(odd_sequences[i][0], odd_sequences[i][1], color='purple', label='Odd Sequence', alpha=0.3)
        #     plt.plot(odd_initial_points[i][0], odd_initial_points[i][1], 'o', color='coral', label='Odd New Initial Point')
        #     plt.plot(odd_cond_points[i][0], odd_cond_points[i][1], 'o', color='lime', label='Odd Cond Initial Point')
        #     plt.annotate(f"{i + 1}", xy=(odd_initial_points[i][0], odd_initial_points[i][1]))
        #     plt.annotate(f"{i + 1}", xy=(odd_cond_points[i][0], odd_cond_points[i][1]))
            
        # for i in range(len(even_sequences)):
        #     plt.plot(even_sequences[i][0], even_sequences[i][1], color='purple', label='Even Sequence')
        #     plt.plot(even_initial_points[i][0], even_initial_points[i][1], 'o', color='crimson', label='Even New Initial Point')
        #     plt.plot(even_cond_points[i][0], even_cond_points[i][1], 'o', color='yellowgreen', label='Even Cond Initial Point')
        #     plt.annotate(f"{2 * i}", xy=(even_initial_points[i][0], even_initial_points[i][1]))
        #     plt.annotate(f"{2 * i}", xy=(even_cond_points[i][0], even_cond_points[i][1]))

        plt.grid()
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.show()


def DoublePreisachModelTest():
    with open(f"../PreisachModel/DoublePreisachModelTest.json") as f:
        data = json.load(f)

        t = np.asarray(data["t"])
        x = np.asarray(data["x"])
        inputs = data["loop"]["in"]
        loop = data["loop"]["out"]

        plot_x(x, t)
        plot_hysteresis_loop(inputs, loop)
        

def article_plots():
    with open(f"../PreisachModel/JustSolveRodos.json") as f:
        data = json.load(f)

        dt = data["dt"]
        time = data["time"]
        t = np.arange(0, time + dt, dt)

        h = data["h"]
        E = data["E"]
        x = np.asarray(data["results"]["x"])
        v = np.asarray(data["results"]["v"])
        inputs = data["loop"]["inputs"]
        loop = data["loop"]["outputs"]

        path = "./article_plots" 
        scientific_plot(x, v, alpha=1, linewidth=.5, xlabel="x(t)", ylabel="v(t)", save=f"{path}/discrete_x2_phase.png")
        scientific_plot(inputs, loop, alpha=1, linewidth=1, xlabel="x(t)", ylabel="P[x(t)]", save=f"{path}/discrete_x2_loop.png")


def OperatorDerivativeTest():
    with open(f"../PreisachModel/OperatorDerivativeTest.json") as f:
        data = json.load(f)
        x = data["results"]["x"]
        derivatives = data["results"]["derivatives"]
        inputs = data["loop"]["inputs"]
        loop = data["loop"]["outputs"]
        
        plt.plot(x)
        plt.plot(derivatives, 'r')
        plt.show()


def ZeroOneTest():
    with open(f"../PreisachModel/ZeroOneTest.json") as f:
        data = json.load(f)
        
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

        plt.plot(param, Ks)
        plt.show()


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
    


if __name__ == '__main__':
    # ZeroOneTest()
    # TestBiffDeidram()
    
    # article_plots()
    # ShuttlePointLog()
    # ShuttlePointTraj()

    # BifurcationDiagram()
    # OperatorDerivativeTest()

    # DoublePreisachModelTest()

    # ShiftTest()

    just_solve_rodos()

