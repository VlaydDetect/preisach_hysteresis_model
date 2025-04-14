import json

import matplotlib.pyplot as plt
import numpy as np
import warnings
from utils import find_local_extrema

from aenum import Flag, auto


class PlottingFlag(Flag):
    AXES = auto()
    POSITION = auto()
    VELOCITY = auto()
    PHASE = auto()
    mLCE_REGRESSION = auto()
    LOOP = auto()
    ALL = AXES | POSITION | VELOCITY | PHASE | mLCE_REGRESSION | LOOP


def draw_axes(ax=None):
    if ax is None:
        ax = plt
    
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')


def plot_x(x, t, ax=None, title=""):
    if ax is None:
        ax = plt

    ax.plot(t, x, label='x(t)')
    draw_axes(ax)
    ax.legend()
    ax.xlabel("Time t")
    ax.ylabel("Position x(t)")
    ax.title(title)
    ax.show()


def plot_v(v, t, ax=None, title=""):
    if ax is None:
        ax = plt

    ax.plot(t, v, label='v(t)')
    draw_axes(ax)
    ax.legend()
    ax.xlabel("Time t")
    ax.ylabel("Velocity v(t)")
    ax.title(title)
    ax.show()


def plot_phase_portrait(x, v, ax=None, title=""):
    if ax is None:
        ax = plt

    ax.plot(x, v)
    draw_axes(ax)
    ax.xlabel("Position x(t)")
    ax.ylabel("Velocity v(t)")
    ax.title(title)
    ax.show()


def plot_mLCE_regression(ns, Ts, ax=None, title=""):
    is_plt = False
    if ax is None:
        is_plt = True
        ax = plt

    ax.scatter(Ts, ns)

    model = np.polyfit(Ts, ns, 1)
    predict = np.poly1d(model)
    x_reg = np.linspace(-2., 1.5)

    ax.plot(x_reg, predict(x_reg), 'r')

    if is_plt:
        ax.title(f"{title}\n Slope (k): {model[0]:.4f}, Intercept (b): {model[1]:.4f}")
    else:
        ax.set_title(f"{title}\n Slope (k): {model[0]:.4f}, Intercept (b): {model[1]:.4f}")


def plot_areaCoeffsLoop(data, flags: PlottingFlag = None):
    if "areaCoeffsLoop" in data:
        areaCoeffsLoop = data["areaCoeffsLoop"]
        M = areaCoeffsLoop["M"]
        e = areaCoeffsLoop["e"]
        Ts = np.asarray(areaCoeffsLoop["Ts"])
        coeffs = areaCoeffsLoop["areaCoeffs"]

        plt.text(.1, .6, f"areaCoeffsLoop: M={M}, e={e}\n Ts={list(np.exp(Ts))}\n areaCoeffs={coeffs}",
                 fontsize=12, clip_on=False)
        plt.show()

        for coeff in coeffs:
            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(25, 10))

            c = areaCoeffsLoop[str(coeff)]
            ns = c["ns"]

            plot_mLCE_regression(ns, Ts, ax=ax[0])

            trajs1 = c["trajs1"]
            trajs2 = c["trajs2"]

            traj1 = np.asarray(trajs1["3.00"])
            traj2 = np.asarray(trajs2["3.00"])
            x1, y1 = [i[0] for i in traj1], [i[1] for i in traj1]
            x2, y2 = [i[0] for i in traj2], [i[1] for i in traj2]

            x11 = np.asarray(x1[:10000])
            x22 = np.asarray(x2[:10000])
            y11 = np.asarray(y1[:10000])
            y22 = np.asarray(y2[:10000])

            ax[1].plot(x11, y11, 'b')
            ax[1].plot(x22, y22, 'r')

            ax[2].plot(np.sqrt(np.square(x11 - x22) + np.square(y11 - y22)))

            fig.suptitle(f"areaCoeffsLoop\n Coeff: {coeff}")
            plt.show()

    else:
        warnings.warn(f"plot_areaCoeffsLoop: areaCoeffsLoop not in file")


def plot_epsilonLoop(data, flags: PlottingFlag = None):
    if "epsilonLoop" in data:
        epsilonLoop = data["epsilonLoop"]
        M = epsilonLoop["M"]
        areaCoeff = epsilonLoop["areaCoeff"]
        Ts = np.asarray(epsilonLoop["Ts"])
        es = epsilonLoop["es"]

        plt.text(.1, .6, f"epsilonLoop: M={M}, areaCoeff={areaCoeff}\n Ts={list(np.exp(Ts))}\n es={es}",
                 fontsize=12, clip_on=False)
        plt.show()

        for e in es:
            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(25, 10))
            c = epsilonLoop[str(e)]
            ns = c["ns"]

            plot_mLCE_regression(ns, Ts, ax=ax[0])

            trajs1 = c["trajs1"]
            trajs2 = c["trajs2"]

            traj1 = np.asarray(trajs1["3.00"])
            traj2 = np.asarray(trajs2["3.00"])
            x1, y1 = [i[0] for i in traj1], [i[1] for i in traj1]
            x2, y2 = [i[0] for i in traj2], [i[1] for i in traj2]

            x11 = np.asarray(x1[:10000])
            x22 = np.asarray(x2[:10000])
            y11 = np.asarray(y1[:10000])
            y22 = np.asarray(y2[:10000])

            ax[1].plot(x11, y11, 'b')
            ax[1].plot(x22, y22, 'r')

            ax[2].plot(np.sqrt(np.square(x11 - x22) + np.square(y11 - y22)))

            fig.suptitle(f"epsilonLoop\n eps: {e}")
            plt.show()

    else:
        warnings.warn(f"plot_epsilonLoop: epsilonLoop not in file")


def plot_hysteresis_loop(inputs, outputs, title=""):
    plt.plot(inputs, outputs, label='loop')

    draw_axes()

    plt.title(title)
    plt.xlabel("Input x")
    plt.ylabel("Output P")
    plt.legend()
    plt.show()


def animate_preisach_plane(uHist, xHist, yHist, outHist, L):
    import matplotlib.animation as animation
    from matplotlib import gridspec

    # historyU = uHist[1:]
    historyU = uHist
    args = [
        historyU,
        xHist,
        yHist,
        outHist
    ]

    def update_lines(frame, args, uLine, stateLine, loopLine):
        print(args)
        uLine.set_xdata([frame])
        uLine.set_ydata([args[0][frame]])

        stateLine.set_xdata([args[1][frame]])
        stateLine.set_ydata([args[2][frame]])

        loopLine.set_xdata([args[0][frame]])
        loopLine.set_ydata([args[3][frame]])

        return uLine, stateLine, loopLine

    frames = len(xHist)

    gs = gridspec.GridSpec(1, 3, height_ratios=[1], width_ratios=[1, 1, 1])
    fig1 = plt.figure(figsize=(18, 5))
    uAx = plt.subplot(gs[0, 0])
    stateAx = plt.subplot(gs[0, 1])
    loopAx = plt.subplot(gs[0, 2])

    # create plot of input
    uAx.plot(uHist, linewidth=1)
    uAx.set_xlim([0, len(uHist)])
    uAx.set_ylim([-L * 1.1, L * 1.1])
    uAx.set_xlabel('samples')
    uAx.set_ylabel('input')
    
    uLine, = uAx.plot([0.0], [0.0], '.', markersize=15)

    # create plot of preisach plane
    stateAx.plot(np.array([-L, L, -L, -L]),
             np.array([-L, L, L, -L]), linewidth=3)
    stateLine, = stateAx.plot([], [], 'r', linewidth=2)
    stateAx.set_xlim(-L * 1.1, L * 1.1)
    stateAx.set_ylim(-L * 1.1, L * 1.1)
    stateAx.set_xlabel('Beta coefficients')
    stateAx.set_ylabel('Alpha coefficients')
    stateAx.legend(['Preisach plane', 'Interface'], loc='lower right')

    # create plot of hysteresis
    loopAx.plot(uHist, outHist)
    loopLine, = loopAx.plot([0.0], [0.0], '.', markersize=15)

    simulation = animation.FuncAnimation(fig1, update_lines, frames,
                                         fargs=(args, uLine, stateLine, loopLine), interval=25,
                                         blit=True, repeat=False)

    plt.show()
    return simulation


def plot_mLCE_diagram(mLCEs, E_vals, h_steps, title="", use_cubic_spline=False):
    from scipy.interpolate import griddata

    x, y, z = np.asarray(h_steps), np.asarray(E_vals), np.asarray(mLCEs)

    warn = False
    if len(x) == len(y):
        x, y = np.meshgrid(x, y)
        if len(z) != len(x):
            warn = True
    else:
        warn = True

    if warn:
        warnings.warn(
            "h_steps and E_vals don't have the same length.\n A linear interpolation grid will be used (it's also "
            "possible to use a cubic one).")
        xi, yi = np.meshgrid(np.linspace(x.min(), x.max(), 100),
                             np.linspace(y.min(), y.max(), 100))
        method = 'cubic' if use_cubic_spline else 'linear'
        zi = griddata((np.repeat(x, len(y)), np.tile(y, len(x))), z, (xi, yi), method=method)
        x, y, z = xi, yi, zi

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(x, y, z, 'g')
    ax.set_title(title)
    ax.set_xlabel('h', fontsize=9)
    ax.set_ylabel('E', fontsize=9)
    ax.set_zlabel('mLCE', fontsize=9)
    plt.show()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(x, y, z, cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel('h', fontsize=9)
    ax.set_ylabel('E', fontsize=9)
    ax.set_zlabel('mLCE', fontsize=9)
    plt.show()


def plot_preisach_derivative(time, inputs, derivatives, title=""):
    plt.plot(time[:-1], inputs, '--', label='input')
    plt.plot(time[:-1], derivatives, 'b', label='Derivative')

    # segments = []
    # segment = []
    #
    # for i in range(len(derivatives)):
    #     # if derivatives[i] == 0:
    #     if abs(derivatives[i]) < 0.05:
    #         if segment:
    #             segments.append(segment)
    #         segment = []
    #     else:
    #         if not segment and i > 0:
    #             segment.append((time[i - 1], 0))
    #         segment.append((time[i], derivatives[i]))
    #
    # if segment:
    #     segments.append(segment)
    #
    # for segment in segments:
    #     sx, sy = zip(*segment)
    #     plt.plot(sx, sy, 'b', label='Derivative')

    draw_axes()

    plt.title(title)
    plt.xlabel("t")
    plt.ylabel("D[u(t)]")
    plt.legend()
    plt.show()


def plot_fourier_transform(sig, dt=None, plot=True):
    # Here it's assumes analytic signal (real signal...) - so only half of the axis is required

    if dt is None:
        dt = 1
        t = np.arange(0, sig.shape[-1])
        xLabel = 'samples'
    else:
        t = np.arange(0, sig.shape[-1]) * dt
        xLabel = 'freq'

    if sig.shape[0] % 2 != 0:
        warnings.warn("signal preferred to be even in size, autoFixing it...")
        t = t[0:-1]
        sig = sig[0:-1]

    sigFFT = np.fft.fft(sig) / t.shape[0]  # Divided by size t for coherent magnitude

    freq = np.fft.fftfreq(t.shape[0], d=dt)

    # Plot analytic signal - right half of frequence axis needed only...
    firstNegInd = np.argmax(freq < 0)
    freqAxisPos = freq[0:firstNegInd]
    sigFFTPos = 2 * sigFFT[0:firstNegInd]  # *2 because of magnitude of analytic signal

    if plot:
        plt.figure()
        plt.plot(freqAxisPos, np.abs(sigFFTPos))
        plt.xlabel(xLabel)
        plt.ylabel('mag')
        plt.title('Analytic FFT plot')
        plt.show()

    return sigFFTPos, freqAxisPos


def plot_bifurcation_diagram(x_values, y_values, x_label, y_label, title=""):
    extrema_x = []
    extrema_y = []

    for x, y in zip(x_values, y_values):
        extremas = find_local_extrema(y_values)
        extrema_x.extend([x] * len(extremas))
        extrema_y.extend(extremas)

    # plt.figure(figsize=(10, 6))
    # plt.scatter(extrema_x, extrema_y, s=10, color='blue')
    plt.plot(extrema_x, extrema_y, 'k.', markersize=1)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if len(title) > 0:
        title = f'Bifurcation diagram ({title})'
    else:
        title = 'Bifurcation diagram'

    plt.title(title)
    plt.grid(True)
    plt.show()


def afc(eps=0.1):
    """
    Compute and plot amplitude-frequency characteristic
    """
    from scipy.signal import find_peaks

    for i in range(1):
        with open(f"../PreisachModel/tests/AFCs/AFC{i}.txt") as f:
            text = f.read()
            data = json.loads(text)

            A = data["A"]

            time = data["time"]
            dt = data["dt"]
            t = np.arange(0.0, time + dt, dt)

            frequencies_inc = np.asarray(data["forward"]["frequencies"])
            frequencies_dec = np.asarray(data["backward"]["frequencies"])

            amplitudes_inc = np.asarray(data["forward"]["amplitudes"])
            amplitudes_dec = np.asarray(data["backward"]["amplitudes"])

            trajs_inc = np.asarray(data["forward"]["trajectories"])
            trajs_dec = np.asarray(data["backward"]["trajectories"])

        plt.plot(frequencies_inc, amplitudes_inc, 'bo--', label="Forward")
        plt.plot(frequencies_dec, amplitudes_dec, 's--', label="Backward")
        plt.xlabel(r'Frequency $\omega$')
        plt.ylabel(r'Amplitude $x_{max}$')
        plt.title(f"A = {A}")
        plt.legend()
        plt.show()
        # plt.savefig(f"./AFCs/AFC{i}.png")

        # for j in range(len(trajs_inc)):
        #     if j % 10 == 0:
        #         plt.plot(t, trajs_inc[j][:, 0], '--', label="Forward")
        #         plt.title(f"A = {A}, Forward Trajectory for w = {frequencies_inc[j]}")
        #         plt.show()
        # 
        #         plt.plot(t, trajs_dec[j][:, 0], '--', label="Backward")
        #         plt.title(f"A = {A}, Backward Trajectory for w = {frequencies_dec[j]}")
        #         plt.show()

    # with open("../PreisachModel/AFCs/AFC0.txt") as f:
    #     text = f.read()
    #     data = json.loads(text)
    #     
    #     A = data["A"]
    # 
    #     frequencies_inc = data["forward"]["frequencies"]
    #     frequencies_dec = data["backward"]["frequencies"]
    # 
    #     amplitudes_inc = data["forward"]["amplitudes"]
    #     amplitudes_dec = data["backward"]["amplitudes"]

    # freqs = data['freqs']
    # signals_inc = data["trajs_inc"]
    # signals_dec = data["trajs_dec"]

    # n = len(freqs)
    # assert n == len(signals_inc)
    # assert n == len(signals_dec)
    # 
    # amplitudes_inc = []
    # frequencies_inc = []
    # amplitudes_dec = []
    # frequencies_dec = []
    # 
    # for i in range(n):
    #     freq = freqs[i]
    #     signal = np.array(signals_inc[i])
    #     signal = signal[:, 0]
    #     peaks, _ = find_peaks(signal)
    #     for j in range(len(peaks) - 1):
    #         if np.abs(signal[peaks[j]] - signal[peaks[j - 1]]) < eps:
    #             amplitude = signal[peaks[-1]]
    #     amplitudes_inc.append(amplitude)
    #     frequencies_inc.append(freq)
    # 
    # freqs = list(reversed(freqs))
    # for i in range(n):
    #     freq = freqs[i]
    #     signal = np.array(signals_dec[i])
    #     signal = signal[:, 0]
    #     peaks, _ = find_peaks(signal)
    #     for j in range(len(peaks) - 1):
    #         if np.abs(signal[peaks[j]] - signal[peaks[j - 1]]) < eps:
    #             amplitude = signal[peaks[-1]]
    #     amplitudes_dec.append(amplitude)
    #     frequencies_dec.append(freq)

    # plt.plot(frequencies_inc, amplitudes_inc, 'bo--', label="Forward")
    # plt.plot(frequencies_dec, amplitudes_dec, 's--', label="Backward")
    # plt.xlabel(r'Frequency $\omega$')
    # plt.ylabel(r'Amplitude $x_{max}$')
    # plt.title(f"A = {A}")
    # plt.legend()
    # plt.show()
