import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

from plotting import scientific_plot


class Args:
    gamma = None
    A = None
    w0 = None
    w = None
    E = None
    eps = None

    def __init__(self, data):
        self.gamma = data["gamma"]
        self.A = data["A"]
        self.w0 = data["w0"]
        self.w = data["w"]
        self.E = data["E"]
        # self.eps = data["eps"]


class BifurcationParams:
    param_name: str
    max_params: np.ndarray
    max_values: np.ndarray
    min_params: np.ndarray
    min_values: np.ndarray

    def __init__(self, data, param_name: str):
        self.param_name = param_name
        self.max_params = np.asarray(data["max_params"])
        self.max_values = np.asarray(data["max_values"])
        self.min_params = np.asarray(data["min_params"])
        self.min_values = np.asarray(data["min_values"])


class Results:
    x: np.ndarray
    v: np.ndarray

    loop = {
        "inputs": np.ndarray,
        "outputs": np.ndarray,
    }
    derivative = np.ndarray

    fourier = {
        "frequencies": np.ndarray,
        "amplitudes": np.ndarray
    }

    biffA: BifurcationParams
    biffE: BifurcationParams
    biffW: BifurcationParams
    biffD: BifurcationParams | None

    LCEs = {
        "LCEs": np.ndarray,
        "history": np.ndarray,
    }
    powerLaw = {
        "e": float,
        "M": int,
        "Ts": np.ndarray,
        "ns": np.ndarray,
        "nu": float,
        "C": float
    }

    poincare = {
        "psm": np.ndarray,
        "pmV": np.ndarray
    }

    afc = {
        "frequencies": np.ndarray,
        "amplitudes_forward": np.ndarray,
        "amplitudes_backward": np.ndarray
    }

    def __init__(self, data0, model):
        data = data0["results"]
        self.x = np.asarray(data["x"])
        self.v = np.asarray(data["v"])

        self.loop = {
            "inputs": np.asarray(data["loop"]["inputs"]),
            "outputs": np.asarray(data["loop"]["outputs"]),
        }
        self.derivative = np.asarray(data["derivatives"])

        self.fourier = {
            "frequencies": np.asarray(data["fourier"]["frequencies"]),
            "amplitudes": np.asarray(data["fourier"]["amplitudes"]),
        }

        self.biffA = BifurcationParams(data["biff"]["A"], "A")
        self.biffE = BifurcationParams(data["biff"]["E"], "E")
        self.biffW = BifurcationParams(data["biff"]["w"], "$\\omega$")
        # TODO: remove after recalc
        self.biffD = BifurcationParams(data0["biff"]["d"], "d") if model != "single_loop" else None

        self.LCEs = {
            "LCEs": np.asarray(data["LCEs"]["LCEs"]),
            "history": np.asarray(data["LCEs"]["history"]),
        }

        self.powerLaw = {
            "e": float(data["LCEs"]["power_law"]["e"]),
            "M": int(data["LCEs"]["power_law"]["M"]),
            "Ts": np.asarray(data["LCEs"]["power_law"]["Ts"]),
            "ns": np.asarray(data["LCEs"]["power_law"]["ns"]),
            "nu": float(data["LCEs"]["power_law"]["nu"]),
            "C": float(data["LCEs"]["power_law"]["C"])
        }

        self.poincare = {
            "psm": np.asarray(data["poincare_mapping"]["psm"]),
            # TODO: remove after recalc
            "pmV": np.asarray(data0["result"]["poincare_mapping"]["pm"]),
        }

        self.afc = {
            "frequencies_forward": np.asarray(data["AFC"]["forward"]["frequencies"]),
            "frequencies_backward": np.asarray(data["AFC"]["backward"]["frequencies"]),
            "amplitudes_forward": np.asarray(data["AFC"]["forward"]["amplitudes"]),
            "amplitudes_backward": np.asarray(data["AFC"]["backward"]["amplitudes"]),
        }


def ComplexRadonsResearch():
    plt.ioff()

    root = Path('./article_plots')
    pattern = "single_loop/*/{0.0, 0.0}/ComplexRadonsResearch.json"

    for file_path in root.glob(pattern):
        try:
            folder_path = file_path.parent
            with file_path.open('r', encoding='utf-8') as f:
                print(f"Process {file_path}")
                data = json.load(f)
                params = data["params"]

                dt = params["dt"]
                time = params["time"]
                model = params["model"]
                args_set = params["args_set"]
                args = Args(params["args"])
                x0 = params["x0"]

                results = Results(data, model)
                t = np.arange(0, time + dt, dt)

                scientific_plot(t, results.x, alpha=1, linewidth=.5, xlabel="t", ylabel="x(t)",
                                save=f"{folder_path}/x.png")
                scientific_plot(results.x, results.v, alpha=1, linewidth=.5, xlabel="x(t)", ylabel="$\\dot{x}(t)$",
                                save=f"{folder_path}/phase.png")
                scientific_plot(results.loop["inputs"], results.loop["outputs"], alpha=1, linewidth=1, xlabel="x(t)",
                                ylabel="P[x(t)]",
                                save=f"{folder_path}/loop.png")

                if model == "single_loop":
                    scientific_plot(t[:-1], results.derivative, alpha=1, linewidth=1, xlabel="t",
                                    ylabel="$\\mathcal{D}_{x(t)} \\mathcal{P}[x(t)]$",
                                    save=f"{folder_path}/derivative.png")

                scientific_plot(results.fourier["frequencies"], results.fourier["amplitudes"], alpha=.5, linewidth=1,
                                xlabel="$\\omega$", ylabel="FFT",
                                save=f"{folder_path}/fourier.png")

                for bif in [results.biffA, results.biffE, results.biffW, results.biffD]:
                    if bif is not None:
                        fig, ax = plt.subplots()
                        ax.scatter(bif.max_params, bif.max_values, c='red', s=0.5, alpha=0.6, label='Local Maxima',
                                   marker='.')
                        ax.scatter(bif.min_params, bif.min_values, c='blue', s=0.5, alpha=0.6, label='Local Minima',
                                   marker='.')

                        ax.set(xlabel=bif.param_name, ylabel='$x_{min}, x_{max}$')
                        ax.legend(markerscale=10)
                        # plt.grid(True, which='both', linestyle='--', alpha=0.5)
                        fig.tight_layout()

                        fig.savefig(f"{folder_path}/bif_{bif.param_name.replace('$\\omega$', 'w')}.png")
                        plt.close(fig)

                fig, ax = plt.subplots()
                ax.plot(results.LCEs["history"][:, 0], c='red')
                ax.plot(results.LCEs["history"][:, 1], c='blue')
                ax.set(xlabel="t", ylabel="$\\lambda$")
                # ax.grid(True, which='both', linestyle='--', alpha=0.5)
                fig.savefig(f"{folder_path}/LCEs.png", dpi=300)
                plt.close(fig)

                fig, ax = plt.subplots()
                ax.scatter(results.powerLaw["Ts"], results.powerLaw["ns"])
                slope, intercept, r_value, _, _ = linregress(results.powerLaw["Ts"], results.powerLaw["Ts"])
                ax.plot(results.powerLaw["Ts"], slope * results.powerLaw["Ts"] + intercept, 'r')
                ax.set(xlabel="T", ylabel="$\\nu$")
                fig.savefig(f"{folder_path}/power_law.png", dpi=300)
                plt.close(fig)

                fig, ax = plt.subplots()
                ax.scatter(results.poincare["psm"][:, 0], results.poincare["psm"][:, 1])
                ax.set(xlabel="x(t)", ylabel="$\\dot{x}(t)$")
                fig.savefig(f"{folder_path}/psm.png", dpi=300)
                plt.close(fig)

                fig, ax = plt.subplots()
                ax.scatter(results.poincare["pmV"][:, 0], results.poincare["pmV"][:, 1])
                ax.set(xlabel="x(t)", ylabel="$\\dot{x}(t)$")
                fig.savefig(f"{folder_path}/pmV.png", dpi=300)
                plt.close(fig)

                fig, ax = plt.subplots()
                ax.plot(results.afc["frequencies_forward"], results.afc["amplitudes_forward"], 'bo--', label="Forward")
                ax.plot(results.afc["frequencies_backward"], results.afc["amplitudes_backward"], 's--',
                        label="Backward")
                ax.set()
                ax.legend()
                fig.savefig(f"{folder_path}/afc.png", dpi=300)
                plt.close(fig)

        except (json.JSONDecodeError, IOError) as e:
            print(f"Ошибка при обработке файла {file_path}: {e}")
