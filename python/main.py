import matplotlib.pyplot as plt
import numpy as np
import glob

from scipy.signal import argrelextrema
from scipy.stats import linregress

from plotting import *
from utils import gen_steps_sequence, select_best_growth
import server


# def f(t, y, gamma, w02, A, w, E):
#     x, v = y
#     dxdt = v
#     dvdt = -gamma * v - w02 * x + A * np.sin(w * t) + E * np.random.random()
#     return [dxdt, dvdt]
#
#
# if __name__ == '__main__':
#     gamma = 0.1
#     w02 = 1.0
#     w = 1.0
#     A = 0.5
#     E_vals = np.linspace(0, 5, 100)
#     y0 = [0.0, 0.0]
#     t_eval = np.linspace(0, 100, 1000)
#
#     extrema = []
#
#     for E in E_vals:
#         sol = solve_ivp(f, t_span=(0, 100), t_eval=t_eval, y0=y0, args=(gamma, w02, A, w, E))
#         x1_sol = sol.y[0]
#
#         maxima = (np.diff(np.sign(np.diff(x1_sol))) < 0).nonzero()[0] + 1
#         minima = (np.diff(np.sign(np.diff(x1_sol))) > 0).nonzero()[0] + 1
#         # print(maxima)
#         # print(minima)
#
#         extrema.append((E, x1_sol[maxima], x1_sol[minima]))
#         print(len(extrema))
#
#         # plt.plot(A * np.ones_like(sol.y[0]), sol.y[0], 'b.', alpha=0.5)
#
#     for data in extrema:
#         E_val, maxima_val, minima_val = data
#         plt.plot([E_val] * len(maxima_val), maxima_val, 'r.', markersize=1)
#         plt.plot([E_val] * len(minima_val), minima_val, 'b.', markersize=1)
#
#     plt.show()


def read_array(file: str, arr):
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            arr.append(float(line))


def discrete_model_chaos_test_by_step_size():
    def gen_sequence(first, n, count):
        result = [first]
        for i in range(1, count):
            first /= n
            result.append(first)
        return result

    h_steps = gen_sequence(0.1, 2, 5)
    for h in h_steps:
        files = glob.glob(fr"../PreisachModel/DiscreteModelChaosTestByStepSize/*_(h={h}).txt")

        E_vals = []
        for file in files:
            start_index = file.find('=')
            end_index = file.find(')')
            val = file[start_index + 1:end_index]
            E_vals.append(float(val))

        x_vals = []
        for file in files:
            x = []
            read_array(file, x)
            x_vals.append(x)

        from utils import find_local_extrema_idx
        for i in range(len(E_vals)):
            E = E_vals[i]
            x = np.array(x_vals[i])
            idx_mins, idx_maxs = find_local_extrema_idx(x)
            mins, maxs = x[idx_mins], x[idx_maxs]
            plt.plot([E] * len(maxs), maxs, 'r.', markersize=2)
            plt.plot([E] * len(mins), mins, 'b.', markersize=2)

        plt.grid(True)
        plt.xlabel("E", fontsize=14)
        plt.ylabel(r'$x_{min}, x_{max}$', fontsize=14)

        plt.title(f"h = {h}")
        plt.show()


# TODO: to type params
def solution_behavior_research_depending_on_h(h_steps=None, auto_params=True, params=None):
    dt = 0.01
    t = np.arange(0, 1800, dt)

    if h_steps is None:
        h_steps = gen_steps_sequence(0.5, 2, 5)

    for h in h_steps:
        files_x = glob.glob(fr"../PreisachModel/tests/x/*_(h={h}).txt")
        files_v = glob.glob(fr"../PreisachModel/tests/v/*_(h={h}).txt")

        x_vals = []
        for file in files_x:
            x = []
            read_array(file, x)
            x_vals.append(x)

        v_vals = []
        for file in files_v:
            v = []
            read_array(file, v)
            v_vals.append(v)

        for i in range(len(x_vals)):
            plot_x(x_vals[i], t, f"h = {h}, E = 1.35")
            plot_v(v_vals[i], t, f"h = {h}, E = 1.35")
            plot_phase_portrait(x_vals[i], v_vals[i], f"h = {h}, E = 1.35")


def lyapunov_exponents_tests_for_oscillator():
    gamma = 0.1
    w02 = 1.
    w = 1.
    A = 0.5
    x0 = np.array([0, 0, 0])
    t0 = 0
    dt = 0.01

    from lyapynov import ContinuousDS
    from lyapynov import mLCE, LCE

    def f(x, t):
        res = np.zeros_like(x)
        res[0] = x[1]
        res[1] = A * np.sin(w * t) - gamma * x[1] - w02 * x[0]
        res[2] = w
        return res

    def jac(x, t):
        res = np.zeros((x.shape[0], x.shape[0]))
        res[0, 1] = 1
        res[1, 0] = -w02
        res[1, 1] = -gamma
        res[1, 2] = A * np.cos(w * t)
        return res

    rodos = ContinuousDS(x0, t0, f, jac, dt)
    rodos.forward(10 ** 6, False)

    mLCE = mLCE(rodos, 0, 10 ** 6, False)
    print(mLCE)


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
        plot_fourier_transform(x, dt=0.01, )

        # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))

        plot_x(x[0:15000], t[0:15000])
        plot_v(v, t)
        plot_phase_portrait(x, v)
        plot_hysteresis_loop(inputs, loop)

        print(loop[:10])
        # plot_preisach_derivative(t, inputs, derivatives)
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
        params = data["A"]

        bifurcation_param = []
        bifurcation_x = []
        
        for i in range(len(params)):
            p = params[i]
            research = trajs[i]
                x = np.asarray([point[0] for point in research])

            maxima = argrelextrema(x, np.greater)
            minima = argrelextrema(x, np.less)
            extremas_idx = np.concatenate([maxima, minima], axis=None)
            x_peaks = x[extremas_idx]

            bifurcation_param.extend([p] * len(x_peaks))
            bifurcation_x.extend(x_peaks)

        plt.figure(figsize=(10, 6))
        plt.plot(bifurcation_param, bifurcation_x, ',k', alpha=1.)
        plt.xlabel('A')
        plt.ylabel('x')
        plt.title('Бифуркационная диаграмма')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
       

if __name__ == '__main__':
    # x = np.linspace(-1, 1, 100)
    # y = np.linspace(-1, 1, 100)
    # X, Y = np.meshgrid(x, y)
    # 
    # def phi(x, a, b, c):
    #     return (a / 2) / (1 + np.cosh((x - b) / c))
    # 
    # def phiphi(x, y, a, b, c):
    #     return phi(x, a, b, c) * phi(-y, a, b, c)
    # 
    # 
    # a = np.asarray([1, 0.5, 0.3])
    # b = np.asarray([1, 0.5, 0.4])
    # c = np.asarray([1, 1, 1])
    # 
    # Z = np.zeros((x.size, y.size))
    # for i in range(len(a)):
    #     Z += phiphi(X, Y, a[i], b[i], c[i])
    # 
    # Z = np.where(X < Y, Z, np.nan)
    # 
    # fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_subplot(111, projection='3d')
    # surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    # 
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.tight_layout()
    # plt.show()

    # just_solve_rodos()

    # plot_everett_fn(X, Y, Z, method='linear')

    # # Примеры временных рядов:
    # # ts1: начинает с роста, затем спад;
    # # ts2: сначала спад, затем резкий рост;
    # # ts3: чёткий рост в начале.
    # ts1 = np.array([1, 2, 3, 2, 1])
    # ts2 = np.array([5, 4, 6, 7, 6])
    # ts3 = np.array([2, 3, 4, 5, 3])
    # 
    # time_series_list = [ts1, ts2, ts3]
    # best_idx, best_segment = select_best_growth(time_series_list)
    # 
    # if best_idx is not None:
    #     print("Выбран временной ряд с индексом:", best_idx)
    #     print("Отрезок роста (start, end, slope):", best_segment)
    # else:
    #     print("Подходящих временных рядов не найдено.")

    # afc()
    # areal_model_test()
    # plot_lce()
    # n_regression()
    # TwoTrajsOnCircle()

    # PoincareMapping()
    BifurcationDiagram()

    # regressionTable()
    # two_trajs()
    # test_trajs_circle()

    # with open(f"../PreisachModel/TEST.json") as f:
    #     data = json.load(f)
    #     plt.plot(data["in"], data["out"])
    #     plt.show()

    # dt = 0.1
    # t = np.arange(0, 600, dt)
    # E = 1.35
    #
    # x = []
    # v = []
    # read_array("../PreisachModel/preisach_x.txt", x)
    # read_array("../PreisachModel/preisach_v.txt", v)
    #
    # inputs = []
    # outputs = []
    # read_array("../PreisachModel/preisach_inputs.txt", inputs)
    # read_array("../PreisachModel/preisach_loop.txt", outputs)
    #
    # plot_x(x, t, E)
    # plot_v(v, t, E)
    # plot_phase_portrait(x, v, E)
    # plot_hysteresis_loop(inputs, outputs, E)
    # plot_fourier_transform(np.array(x), dt)

    dt = 0.01
    t = np.arange(0, 1800, dt)
    # s = []
    # read_array("../PreisachModel/preisach_sin.txt", s)
    #
    # plot_x(s, t, 0.0)
    #
    # sinus = 2 * np.sin(t)
    # plt.plot(t, sinus)
    # plt.show()

    # inputs = []
    # outputs = []
    # read_array("../PreisachModel/preisach_inputs.txt", inputs)
    # read_array("../PreisachModel/preisach_loop.txt", outputs)
    # plot_hysteresis_loop(inputs, outputs, 0.0)

    # x = []
    # v = []
    # read_array("../PreisachModel/preisach_x_E=3.0.txt", x)
    # read_array("../PreisachModel/preisach_v_E=3.0.txt", v)
    #
    # plot_x(x, t, 3.0)
    # plot_v(v, t, 3.0)
    # plot_phase_portrait(x, v, 3.0)

    # discrete_model_chaos_test_by_step_size()

    # python_server = server.Server()
    # python_server.start()

    # import json
    # from plotting import plot_mLCE_diagram
    # h_steps = [0.1, 0.05, 0.025, 0.0125]
    # E_vals = np.arange(0.5, 2.5, 0.1)
    # mLCEs = []
    # files = glob.glob(fr"../PreisachModel/tests/LCEs/*.txt")
    # for file in files:
    #     with open(file) as f:
    #         text = f.read()
    #         data = json.loads(text)
    #         mLCEs.append(data["mLCE"])
    # 
    # plot_mLCE_diagram(mLCEs, E_vals, h_steps, "", True)

    # import json
    # 
    # with open("../PreisachModel/trajs.txt") as f:
    #     text = f.read()
    #     data = json.loads(text)
    # 
    #     traj1 = data["traj1"]
    #     traj2 = data["traj2"]
    #     
    #     loop1 = data["loop1"]
    #     loop2 = data["loop2"]
    # 
    #     x1, y1 = [i[0] for i in traj1], [i[1] for i in traj1]
    #     t1 = np.arange(0, len(y1) * 0.01, 0.01)
    #     x2, y2 = [i[0] for i in traj2], [i[1] for i in traj2]
    #     t2 = np.arange(0, len(y2) * 0.01, 0.01)
    # 
    #     fig, axs = plt.subplots(2)
    #     fig.suptitle("Two close trajectories")
    #     axs[0].plot(x1, y1, 'b')
    #     axs[1].plot(x2, y2, 'r')
    # 
    #     n = 1000
    # 
    #     x1, y1 = x1[:n], y1[:n]
    #     x2, y2 = x2[:n], y2[:n]
    #     tn = t[:n]
    # 
    #     fig, axs = plt.subplots(2)
    #     fig.suptitle("Two close trajectories")
    #     # axs[0].plot(x1, y1, 'b')
    #     # axs[1].plot(x2, y2, 'r')
    #     axs[0].plot(x1, y1, 'b', label='Trajectory 1')
    #     axs[1].plot(x2, y2, 'r', label='Trajectory 2')
    #     fig.legend()
    #     plt.show()
    # 
    #     # plt.plot(x1, y1, 'b')
    #     # plt.plot(x2, y2, 'r')
    #     plt.plot(x1, y1, 'b', label='Trajectory 1')
    #     plt.plot(x2, y2, 'r', label='Trajectory 2')
    #     plt.legend()
    #     plt.show()

    # fig, axs = plt.subplots(2)
    # fig.suptitle("Two loops")
    # axs[0].plot(loop1["in"], loop1["out"], 'b')
    # axs[1].plot(loop2["in"], loop2["out"], 'r')
    # plt.show()

    # plot_mLCE_diagram(data["mLCEs"], data["E_vals"], data["h_steps"], data["name"])
