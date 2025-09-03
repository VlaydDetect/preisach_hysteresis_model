import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, argrelextrema

if __name__ == "__main__":

    # Параметры
    sigma = 10.
    beta = 8. / 3.
    rho_values = np.linspace(16., 200., 200)

    # Временной интервал
    t_span = (0, 100)
    t_eval = np.linspace(*t_span, 10000)

    # Начальные условия
    y0 = [.1, -10., 0.1]

    # Хранение результата
    bifurcation_rho = []
    bifurcation_z = []


    def lorenz(t, state, sigma, beta, rho):
        x, y, z = state
        dxdt = sigma * (y - x)
        dydt = x * (rho - z) - y
        dzdt = x * y - beta * z
        return [dxdt, dydt, dzdt]


    for rho in rho_values:
        sol = solve_ivp(
            lorenz, t_span, y0, args=(sigma, beta, rho),
            t_eval=t_eval, rtol=1e-8, atol=1e-10
        )

        x = sol.y[0]
        y = sol.y[1]
        z = sol.y[2]
        
        # plt.plot(t_eval, x)
        # plt.title(f'x(t), rho={rho}')
        # plt.grid(True)
        # plt.show()

        # ax = plt.figure().add_subplot(projection='3d')
        # ax.plot(x, y, z, lw=0.5)
        # ax.set_xlabel("X")
        # ax.set_ylabel("Y")
        # ax.set_zlabel("Z")
        # ax.set_title(f'3D, rho={rho}')
        # ax.grid(True)
        # plt.show()
        
        # Удаляем начальную часть
        transient = int(len(z) * 0.5)
        z = z[transient:]

        # Находим пики
        # peaks, _ = find_peaks(z, distance=10)
        # z_peaks = z[peaks]

        # peaks_max, _ = find_peaks(z, distance=10)
        # peaks_min, _ = find_peaks(-z, distance=10)
        # all_peaks = np.concatenate([peaks_max, peaks_min])
        # z_extrema = z[all_peaks]
        # sorted_indices = np.argsort(all_peaks)
        # z_peaks = z_extrema[sorted_indices]
        maxima = argrelextrema(z, np.greater)
        minima = argrelextrema(z, np.less)
        extremas_idx = np.concatenate([maxima, minima], axis=None)
        z_peaks = z[extremas_idx]

        # Добавляем точки на диаграмму
        bifurcation_rho.extend([rho] * len(z_peaks))
        bifurcation_z.extend(z_peaks)

    # Построение
    plt.figure(figsize=(10, 6))
    plt.plot(bifurcation_rho, bifurcation_z, ',k', alpha=0.5)
    plt.xlabel(r'$\rho$')
    plt.ylabel(r'$z$ (локальные максимумы)')
    plt.title('Бифуркационная диаграмма системы Лоренца по параметру $\\rho$')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
