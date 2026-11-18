"""Interactive 2D / 3D Preisach-density explorer based on the Szabo/Fuzi kernels.

Requirements:
    pip install numpy matplotlib

Run on a desktop, with an interactive Matplotlib GUI backend:
    python preisach_density_explorer.py

The plotted function is the density mu(beta, alpha), not the Everett integral E.
The physical support is beta <= alpha. Parameter arrays A0, B0, C0 support
1 to 5 components. Amplitudes a_i have logarithmic sliders: a_i = 10**s_i.

Controls:
    - click "3D поверхность" / "2D теплокарта" to switch visualizations;
    - in 3D use the left mouse button to rotate, scroll to zoom;
    - sliders and checkboxes work in either mode;
    - buttons export the CURRENT view as PNG and the parameters as JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.tri import Triangulation
from matplotlib.widgets import Button, CheckButtons, Slider
import numpy as np


# -------------------- User configuration --------------------
BOUNDS = (-1.0, 1.0)
RESOLUTION = 280             # 2D image & integration grid
RESOLUTION_3D = 44          # Fast 3D triangular mesh; raise to ~80 for extra detail

# The parameter arrays must all have the same length (from 1 to 5).
A0 = np.array([4299.43437, 5.0, 60.0], dtype=float)
B0 = np.array([1.5, 0.0, 1.2], dtype=float)
C0 = np.array([0.2, 0.25, 0.3], dtype=float)
ENABLED0 = [True, False, False]

LOG10_A_RANGE = (-5.0, 6.0)
B_RANGE = (-3.0, 3.0)
C_RANGE = (0.02, 1.5)

NORMALIZE_MASS_BY_DEFAULT = False
AUTOSCALE_COLORS_BY_DEFAULT = True
START_IN_3D = False
DEFAULT_CAMERA = (14.0, -141.0, 0.0)  # elev, azim, roll
# ------------------------------------------------------------


def phi(t: np.ndarray | float, a: float, b: float, c: float) -> np.ndarray:
    """phi(t) = (a/2)/(1+cosh((t-b)/c)), evaluated without cosh overflow."""
    z = np.abs((np.asarray(t) - b) / c)
    e = np.exp(-z)
    return a * e / np.square(1.0 + e)


class PreisachExplorer:
    """Adjust a nonnegative Preisach density on the triangular support."""

    def __init__(self) -> None:
        self.a = np.array(A0, dtype=float, copy=True)
        self.b = np.array(B0, dtype=float, copy=True)
        self.c = np.array(C0, dtype=float, copy=True)
        self.n = len(self.a)

        if not (1 <= self.n <= 5):
            raise ValueError("Use 1 to 5 components for this slider layout")
        if not (len(self.b) == len(self.c) == len(ENABLED0) == self.n):
            raise ValueError("A0, B0, C0 and ENABLED0 must have the same length")
        if not np.all(np.isfinite(self.a)) or np.any(self.a <= 0):
            raise ValueError("All a_i must be finite and > 0 (logarithmic sliders)")
        if not np.all(np.isfinite(self.b)) or not np.all(np.isfinite(self.c)):
            raise ValueError("b_i and c_i must be finite")
        if np.any(self.c <= 0):
            raise ValueError("All c_i must be > 0")
        if not BOUNDS[0] < BOUNDS[1] or RESOLUTION < 2 or RESOLUTION_3D < 3:
            raise ValueError("Invalid bounds or grid resolution")
        if not all(LOG10_A_RANGE[0] <= np.log10(x) <= LOG10_A_RANGE[1] for x in self.a):
            raise ValueError("Initial amplitudes outside LOG10_A_RANGE")
        if not all(B_RANGE[0] <= x <= B_RANGE[1] for x in self.b):
            raise ValueError("Initial b values outside B_RANGE")
        if not all(C_RANGE[0] <= x <= C_RANGE[1] for x in self.c):
            raise ValueError("Initial c values outside C_RANGE")

        self.enabled = list(ENABLED0)
        self.normalize_mass = NORMALIZE_MASS_BY_DEFAULT
        self.autoscale = AUTOSCALE_COLORS_BY_DEFAULT
        self.is_3d = START_IN_3D
        self._updating_controls = False

        lo, hi = BOUNDS
        self.h = (hi - lo) / RESOLUTION
        # Centers for midpoint quadrature; diagonal cells count half their area.
        self.grid = lo + (np.arange(RESOLUTION) + 0.5) * self.h
        self.inside = np.tri(RESOLUTION, RESOLUTION, dtype=bool)  # row=alpha, col=beta
        self.integration_weights = self.inside.astype(float)
        np.fill_diagonal(self.integration_weights, 0.5)
        self.target_mass = 0.5 * (hi - lo) ** 2

        # Use a separate triangulated grid in 3D with EXACT diagonal boundary,
        # rather than plotting a rectangular surface with NaN holes.
        self.grid_3d = np.linspace(lo, hi, RESOLUTION_3D)
        self.triangles_3d = self._build_triangles(RESOLUTION_3D)
        xb, ya = np.meshgrid(self.grid_3d, self.grid_3d)
        self.triangulation_3d = Triangulation(
            xb.ravel(), ya.ravel(), triangles=self.triangles_3d
        )

        self.fig = plt.figure(figsize=(15.6, 9.0), facecolor="#fafafa")
        if hasattr(self.fig.canvas.manager, "set_window_title"):
            self.fig.canvas.manager.set_window_title("Preisach density explorer: 2D / 3D")

        self.plot_rect = (0.055, 0.16, 0.505, 0.76)
        self.cmap = plt.get_cmap("viridis").copy()
        self.cmap.set_bad(color="#fafafa")
        self.norm = Normalize(vmin=0.0, vmax=1.0)
        self.color_mappable = ScalarMappable(norm=self.norm, cmap=self.cmap)
        self.cbar_ax = self.fig.add_axes((0.592, 0.23, 0.020, 0.60))
        self.cbar = self.fig.colorbar(self.color_mappable, cax=self.cbar_ax)
        self.cbar.set_label(r"Плотность $\mu(\beta,\alpha)$")

        self.ax = None
        self.image = None
        self.surface = None
        self.camera = DEFAULT_CAMERA

        self.info_line1 = self.fig.text(0.055, 0.089, "", fontsize=11, color="#292929")
        self.info_line2 = self.fig.text(0.055, 0.057, "", fontsize=10, color="#555555")
        self.status_line = self.fig.text(0.66, 0.013, "", fontsize=9, color="#444444")

        self.sliders: list[tuple[Slider, Slider, Slider]] = []
        top, bottom = 0.913, 0.337
        group_step = (top - bottom) / self.n

        self.fig.text(0.66, 0.966, "Параметры компонент", fontsize=15, weight="bold")
        self.fig.text(0.66, 0.939, r"$a_i=10^{\mathrm{slider}}$  (логарифмическая шкала)", fontsize=10)

        for i in range(self.n):
            heading_y = top - i * group_step
            self.fig.text(0.662, heading_y, f"Компонента {i + 1}", fontsize=11, weight="bold")
            sliders_i = []
            for j, (name, rng, initial) in enumerate((
                (r"$\log_{10}a$", LOG10_A_RANGE, float(np.log10(self.a[i]))),
                (r"$b$", B_RANGE, float(self.b[i])),
                (r"$c$", C_RANGE, float(self.c[i])),
            )):
                y = heading_y - group_step * (0.24 + 0.23 * j)
                slider_ax = self.fig.add_axes((0.716, y, 0.215, 0.017), facecolor="#ebebeb")
                slider = Slider(slider_ax, name, rng[0], rng[1], valinit=initial, valfmt="%.3f")
                sliders_i.append(slider)
            self.sliders.append(tuple(sliders_i))

        # One-click view switch; unlike re-creating the whole figure, this keeps
        # ALL sliders, switches, buttons and the colorbar alive.
        view_ax = self.fig.add_axes((0.663, 0.324, 0.160, 0.036))
        self.view_button = Button(view_ax, "", color="#eaeaea", hovercolor="#d9e7fa")
        camera_ax = self.fig.add_axes((0.835, 0.324, 0.132, 0.036))
        self.camera_button = Button(camera_ax, "Камера: сброс")
        self.camera_button.ax.set_visible(self.is_3d)

        check_labels = [f"Включить компоненту {i+1}" for i in range(self.n)]
        check_labels.extend(["Нормировать массу к площади", "Автомасштаб цветов"])
        check_values = self.enabled + [self.normalize_mass, self.autoscale]
        checkbox_ax = self.fig.add_axes((0.661, 0.103, 0.306, 0.210), facecolor="#fafafa")
        checkbox_ax.set_axis_off()
        self.checks = CheckButtons(checkbox_ax, check_labels, check_values)
        for label in self.checks.labels:
            label.set_fontsize(9)

        reset_ax = self.fig.add_axes((0.663, 0.045, 0.090, 0.036))
        json_ax = self.fig.add_axes((0.768, 0.045, 0.095, 0.036))
        png_ax = self.fig.add_axes((0.875, 0.045, 0.095, 0.036))
        self.reset_button = Button(reset_ax, "Сброс")
        self.json_button = Button(json_ax, "JSON")
        self.png_button = Button(png_ax, "PNG")

        self.scale = 1.0
        self.mass_raw = 0.0
        self.mass_shown = 0.0
        self.fixed_vmax: float | None = None
        self.last_shown = np.zeros((RESOLUTION, RESOLUTION), dtype=float)
        self.last_max = 0.0

        self._switch_plot_axes()
        # Initialize before connecting callbacks; avoids unnecessary redraws.
        self.update()
        self.fixed_vmax = float(self.norm.vmax)

        for trio in self.sliders:
            for slider in trio:
                slider.on_changed(self.on_parameter_changed)
        self.checks.on_clicked(self.on_check_changed)
        self.view_button.on_clicked(self.on_toggle_view)
        self.camera_button.on_clicked(self.on_reset_camera)
        self.reset_button.on_clicked(self.on_reset)
        self.json_button.on_clicked(self.on_save_json)
        self.png_button.on_clicked(self.on_save_png)

    @staticmethod
    def _build_triangles(n: int) -> np.ndarray:
        """Triangle connectivity for alpha >= beta, including the diagonal."""
        triangles = []
        for row in range(n - 1):  # alpha index
            for col in range(row + 1):  # beta index
                v00 = row * n + col
                v10 = (row + 1) * n + col
                v11 = (row + 1) * n + col + 1
                if col == row:
                    # Exactly half of the diagonal square is in the support.
                    triangles.append((v00, v11, v10))
                else:
                    v01 = row * n + col + 1
                    triangles.append((v00, v01, v11))
                    triangles.append((v00, v11, v10))
        return np.asarray(triangles, dtype=np.int32)

    def _switch_plot_axes(self) -> None:
        """Switch Matplotlib axes without destroying the control panel."""
        if self.ax is not None:
            if hasattr(self.ax, "elev"):
                self.camera = (self.ax.elev, self.ax.azim, self.ax.roll)
            self.ax.remove()

        self.ax = self.fig.add_axes(
            self.plot_rect, projection="3d" if self.is_3d else None
        )
        self.image = None
        self.surface = None
        lo, hi = BOUNDS

        if self.is_3d:
            self.ax.set_xlim(lo, hi)
            self.ax.set_ylim(lo, hi)
            self.ax.set_xlabel(r"$\beta$ — нижний порог", labelpad=9)
            self.ax.set_ylabel(r"$\alpha$ — верхний порог", labelpad=9)
            self.ax.set_zlabel(r"$\mu(\beta,\alpha)$", labelpad=4)
            self.ax.set_title("Плотность Прейзаха: 3D", fontsize=15, pad=12)
            self.ax.set_box_aspect((1.0, 1.0, 0.67))
            self.ax.view_init(elev=self.camera[0], azim=self.camera[1], roll=self.camera[2])
            self.ax.ticklabel_format(axis="z", style="sci", scilimits=(0, 0))
        else:
            self.image = self.ax.imshow(
                np.ma.masked_array(
                    np.zeros((RESOLUTION, RESOLUTION)), mask=~self.inside
                ),
                origin="lower", extent=(lo, hi, lo, hi),
                interpolation="nearest", aspect="equal",
                cmap=self.cmap, norm=self.norm,
            )
            self.ax.plot([lo, hi], [lo, hi], color="#181818", linewidth=0.9, alpha=0.75)
            self.ax.set_xlabel(r"$\beta$ — нижний порог")
            self.ax.set_ylabel(r"$\alpha$ — верхний порог")
            self.ax.set_title(r"Плотность Прейзаха $\mu(\beta,\alpha)$: 2D", fontsize=15, pad=15)
            self.ax.set_xlim(lo, hi)
            self.ax.set_ylim(lo, hi)

        self.camera_button.ax.set_visible(self.is_3d)
        self.view_button.label.set_text("2D теплокарта" if self.is_3d else "3D поверхность")

    def _density(self, grid: np.ndarray) -> np.ndarray:
        """mu(beta, alpha) = sum phi_i(beta)*phi_i(-alpha); row=alpha."""
        values = np.zeros((grid.size, grid.size), dtype=float)
        for i, enabled in enumerate(self.enabled):
            if enabled:
                p_beta = phi(grid, self.a[i], self.b[i], self.c[i])
                p_neg_alpha = phi(-grid, self.a[i], self.b[i], self.c[i])
                values += np.outer(p_neg_alpha, p_beta)
        return values

    def on_parameter_changed(self, _new_value: float) -> None:
        if self._updating_controls:
            return
        for i, (aslider, bslider, cslider) in enumerate(self.sliders):
            self.a[i] = 10.0 ** aslider.val
            self.b[i] = bslider.val
            self.c[i] = cslider.val
        self.update()

    def on_check_changed(self, label: str) -> None:
        if self._updating_controls:
            return
        if label.startswith("Включить компоненту"):
            i = int(label.split()[-1]) - 1
            self.enabled[i] = not self.enabled[i]
        elif label == "Нормировать массу к площади":
            self.normalize_mass = not self.normalize_mass
        elif label == "Автомасштаб цветов":
            self.autoscale = not self.autoscale
        self.update()

    def on_toggle_view(self, _event) -> None:
        self.is_3d = not self.is_3d
        self._switch_plot_axes()
        self.update()
        self.status_line.set_text(
            "3D: ЛКМ — вращать, колёсико — масштаб"
            if self.is_3d else "Режим 2D"
        )
        self.fig.canvas.draw_idle()

    def on_reset_camera(self, _event) -> None:
        if not self.is_3d:
            return
        self.camera = DEFAULT_CAMERA
        self.ax.view_init(elev=self.camera[0], azim=self.camera[1], roll=self.camera[2])
        self.fig.canvas.draw_idle()

    def on_reset(self, _event) -> None:
        self._updating_controls = True
        try:
            for i, trio in enumerate(self.sliders):
                for slider, value in zip(trio, (np.log10(A0[i]), B0[i], C0[i])):
                    slider.set_val(float(value))
            desired = list(ENABLED0) + [NORMALIZE_MASS_BY_DEFAULT, AUTOSCALE_COLORS_BY_DEFAULT]
            for i, (actual, target) in enumerate(zip(self.checks.get_status(), desired)):
                if actual != target:
                    self.checks.set_active(i)
        finally:
            self._updating_controls = False
        self.a = np.array(A0, dtype=float, copy=True)
        self.b = np.array(B0, dtype=float, copy=True)
        self.c = np.array(C0, dtype=float, copy=True)
        self.enabled = list(ENABLED0)
        self.normalize_mass = NORMALIZE_MASS_BY_DEFAULT
        self.autoscale = AUTOSCALE_COLORS_BY_DEFAULT
        self.update()
        self.status_line.set_text("Настройки сброшены")
        self.fig.canvas.draw_idle()

    def update(self) -> None:
        density = self._density(self.grid)
        corner = sum(
            float(phi(BOUNDS[0], self.a[i], self.b[i], self.c[i]) *
                  phi(-BOUNDS[1], self.a[i], self.b[i], self.c[i]))
            for i, enabled in enumerate(self.enabled) if enabled
        )
        diagonal_center = sum(
            float(phi(0.0, self.a[i], self.b[i], self.c[i]) ** 2)
            for i, enabled in enumerate(self.enabled) if enabled
        )

        # Numerical integration for the mass; no integration is needed
        # to calculate the density itself.
        self.mass_raw = float(np.sum(density * self.integration_weights) * self.h ** 2)
        self.scale = 1.0
        if self.normalize_mass and self.mass_raw > 0.0 and np.isfinite(self.mass_raw):
            self.scale = self.target_mass / self.mass_raw
        elif self.normalize_mass:
            self.status_line.set_text("Нормировка невозможна: нулевая масса")

        self.mass_shown = self.mass_raw * self.scale
        shown = density * self.scale
        self.last_shown = shown
        self.last_max = float(shown[self.inside].max(initial=0.0))
        if self.autoscale or self.fixed_vmax is None:
            color_max = self.last_max if self.last_max > 0.0 else 1.0
        else:
            color_max = self.fixed_vmax

        # One consistent color scale in both modes and a persistent colorbar.
        self.color_mappable.set_clim(0.0, color_max)
        if not self.is_3d:
            self.image.set_data(np.ma.masked_array(shown, mask=~self.inside))
        else:
            if self.surface is not None:
                self.surface.remove()

            surface_z = (self._density(self.grid_3d) * self.scale).ravel()
            self.surface = self.ax.plot_trisurf(
                self.triangulation_3d,
                surface_z,
                cmap=self.cmap,
                norm=self.norm,
                linewidth=0.0,
                antialiased=False,
                shade=False,
            )
            self.ax.set_zlim(0.0, self.last_max * 1.05 if self.last_max > 0.0 else 1.0)

        for a_slider, _, _ in self.sliders:
            a_slider.valtext.set_text(f"a={10 ** a_slider.val:.4g}")

        self.info_line1.set_text(
            f"Масса носителя  ∫ μ = {self.mass_shown:.6g}  "
            f"(без нормировки: {self.mass_raw:.6g});   коэффициент = {self.scale:.5g}"
        )
        self.info_line2.set_text(
            f"μ(-1, 1) = {corner * self.scale:.5g};   "
            f"μ(0, 0) = {diagonal_center * self.scale:.5g};   "
            f"max μ = {self.last_max:.5g}"
        )
        self.fig.canvas.draw_idle()

    def on_save_json(self, _event) -> None:
        # Multiplying a_i by sqrt(scale) multiplies both mu and Everett by scale.
        effective_a = (self.a * np.sqrt(self.scale)).tolist()
        payload = {
            "description": "mu(beta, alpha) = sum_i phi_i(beta) phi_i(-alpha), beta <= alpha",
            "bounds": list(BOUNDS),
            "a": self.a.tolist(),
            "b": self.b.tolist(),
            "c": self.c.tolist(),
            "enabled": self.enabled,
            "normalize_mass": self.normalize_mass,
            "normalization_scale": self.scale,
            "effective_a": effective_a,
            "surface_mass": self.mass_shown,
            "view_mode": "3D" if self.is_3d else "2D",
            "camera_3d": {
                "elev": float(self.ax.elev) if self.is_3d else float(self.camera[0]),
                "azim": float(self.ax.azim) if self.is_3d else float(self.camera[1]),
                "roll": float(self.ax.roll) if self.is_3d else float(self.camera[2]),
            },
        }
        path = Path("preisach_parameters.json").resolve()
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        self.status_line.set_text(f"Сохранено: {path.name}")
        self.fig.canvas.draw_idle()

    def on_save_png(self, _event) -> None:
        path = Path("preisach_density.png").resolve()
        self.fig.savefig(path, dpi=160, facecolor=self.fig.get_facecolor())
        self.status_line.set_text(f"Сохранено: {path.name}")
        self.fig.canvas.draw_idle()


def main() -> None:
    explorer = PreisachExplorer()
    plt.show()


if __name__ == "__main__":
    main()
