import numpy as np
import matplotlib.pyplot as plt
import warnings


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


def plot_x(x, t, E_val):
    plt.plot(t, x, label='x(t)')
    plt.legend()
    plt.xlabel("Time t")
    plt.ylabel("Position x(t)")
    plt.title(f"E = {E_val}")
    plt.show()


def plot_v(v, t, E_val):
    plt.plot(t, v, label='v(t)')
    plt.legend()
    plt.xlabel("Time t")
    plt.ylabel("Velocity x(t)")
    plt.title(f"E = {E_val}")
    plt.show()


def plot_phase_portrait(x, v, E_val):
    plt.plot(x, v)
    plt.xlabel("Position x(t)")
    plt.ylabel("Velocity v(t)")
    plt.title(f"E = {E_val}")
    plt.show()


def plot_hysteresis_loop(inputs, outputs, E_val):
    plt.plot(inputs, outputs, label='loop')

    ax = plt.gca()
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')

    plt.title(f"E = {E_val}")
    plt.xlabel("Input x")
    plt.ylabel("Output P")
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


if __name__ == '__main__':
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

    dt = 0.1
    t = np.arange(0, 100, dt)
    # s = []
    # read_array("../PreisachModel/preisach_sin.txt", s)
    #
    # plot_x(s, t, 0.0)
    #
    # sinus = 2 * np.sin(t)
    # plt.plot(t, sinus)
    # plt.show()

    inputs = []
    outputs = []
    read_array("../PreisachModel/preisach_inputs.txt", inputs)
    read_array("../PreisachModel/preisach_loop.txt", outputs)
    plot_hysteresis_loop(inputs, outputs, 0.0)
