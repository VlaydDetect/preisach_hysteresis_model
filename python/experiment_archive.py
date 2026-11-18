import argparse
import json
import math
import re
import zipfile

import numpy as np


PLOT_INPUTS = {
    "displacement": ("x", "time"),
    "velocity": ("v", "time"),
    "phase_portrait": ("x", "v"),
    "hysteresis_loop": ("loop_inputs", "loop_outputs"),
    "fourier_spectrum": ("x",),
    "bifurcation": ("max_params", "max_values", "min_params", "min_values"),
    "zero_one": ("amplitudes", "ks"),
    "divergence_regression": ("mean_log_distances", "log_times"),
    "power_law": ("power_mean_log_distances", "power_log_times"),
    "phase_pair": ("primary", "perturbed"),
    "directional_phase_pair": ("primary_0", "perturbed_0"),
}
_LABELS = {"title", "x_label", "y_label"}
_IDENTIFIER = re.compile(r"[a-z0-9_]+")
_MAX_MANIFEST_BYTES = 8 * 1024 * 1024
_CHUNK_BYTES = 256 * 1024


def _identifier(value):
    return isinstance(value, str) and _IDENTIFIER.fullmatch(value) is not None


def _array_path(name):
    return f"arrays/{name}.npy"


def _resolve_results(value, arrays):
    if isinstance(value, dict):
        if "$array" in value:
            name = value["$array"]
            if set(value) != {"$array"} or not _identifier(name) or name not in arrays:
                raise ValueError(f"Invalid array reference: {value!r}")
            return arrays[name]
        return {key: _resolve_results(item, arrays) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_results(item, arrays) for item in value]
    return value


def _validate_manifest(manifest):
    if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int:
        raise ValueError("Missing archive schema version")
    if manifest["schema_version"] != 1:
        raise ValueError(f"Unsupported archive schema version {manifest['schema_version']}")
    if not _identifier(manifest.get("experiment_kind")):
        raise ValueError("Invalid experiment kind")
    parameters = manifest.get("parameters")
    arrays = manifest.get("arrays")
    plots = manifest.get("plots")
    if not isinstance(parameters, dict) or not isinstance(arrays, dict) or not isinstance(plots, list):
        raise ValueError("Expected parameters, arrays and plots in the manifest")

    for name, info in arrays.items():
        if not _identifier(name) or not isinstance(info, dict):
            raise ValueError(f"Invalid array entry {name!r}")
        shape, axes, dtype = info.get("shape"), info.get("axes"), info.get("dtype")
        if (not isinstance(shape, list) or not shape
                or any(type(size) is not int or size < 0 for size in shape)
                or not isinstance(axes, list) or len(axes) != len(shape)
                or any(not isinstance(axis, str) or not axis for axis in axes)
                or not isinstance(dtype, str)):
            raise ValueError(f"Invalid array metadata for {name}")
        if dtype not in {"<f4", "<f8", "|i1", "|u1", "<i2", "<u2", "<i4", "<u4", "<i8", "<u8"}:
            raise ValueError(f"Unsupported numerical dtype for {name}: {dtype}")

    if "results" in manifest:
        if not isinstance(manifest["results"], dict):
            raise ValueError("Expected a results object in the manifest")
        _resolve_results(manifest["results"], arrays)

    for plot in plots:
        if not isinstance(plot, dict) or not _identifier(plot.get("kind")) or plot["kind"] not in PLOT_INPUTS:
            raise ValueError(f"Unsupported plot: {plot!r}")
        if set(plot) - _LABELS - {"kind"} or any(not isinstance(plot[name], str) for name in _LABELS & plot.keys()):
            raise ValueError(f"Invalid plot options for {plot['kind']}")
        names = PLOT_INPUTS[plot["kind"]]
        if any(name not in arrays for name in names):
            raise ValueError(f"Missing data for {plot['kind']}")
        shapes = [arrays[name]["shape"] for name in names]
        if plot["kind"] in {"phase_pair", "directional_phase_pair"}:
            valid_shapes = (all(len(shape) == 2 and shape[1] >= 2 for shape in shapes)
                            and shapes[0][1] == shapes[1][1])
        elif plot["kind"] == "bifurcation":
            valid_shapes = all(len(shape) == 1 for shape in shapes) and shapes[0] == shapes[1] and shapes[2] == shapes[3]
        else:
            valid_shapes = all(len(shape) == 1 and shape == shapes[0] for shape in shapes)
        if not valid_shapes:
            raise ValueError(f"Incompatible array shapes for {plot['kind']}")
        if plot["kind"] == "fourier_spectrum":
            dt = parameters.get("dt")
            if type(dt) not in (int, float) or not math.isfinite(dt) or dt <= 0 or shapes[0][0] < 5:
                raise ValueError("Fourier spectrum requires a positive step and at least five samples")


def read_archive(path, *, load_arrays=True):
    """Read a result, or validate it without allocating the numerical arrays."""
    with zipfile.ZipFile(path) as archive:
        entries = {entry.filename: entry for entry in archive.infolist()}
        if len(entries) != len(archive.infolist()):
            raise ValueError("Duplicate entries in experiment archive")
        if any(entry.compress_type != zipfile.ZIP_STORED or entry.flag_bits & 1
               or entry.compress_size != entry.file_size for entry in entries.values()):
            raise ValueError("Archive entries must be stored as unencrypted files")
        manifest_entry = entries.get("manifest.json")
        if manifest_entry is None or manifest_entry.file_size > _MAX_MANIFEST_BYTES:
            raise ValueError("Missing or oversized manifest")
        manifest = json.loads(archive.read(manifest_entry))
        _validate_manifest(manifest)
        expected = {"manifest.json"} | {_array_path(name) for name in manifest["arrays"]}
        if entries.keys() != expected:
            raise ValueError("Archive entries do not match the manifest")

        selected = set(manifest["arrays"]) if load_arrays is True else set()
        if load_arrays == "plots":
            selected = {name for plot in manifest["plots"] for name in PLOT_INPUTS[plot["kind"]]}
        arrays = {}
        for name, info in manifest["arrays"].items():
            entry = entries[_array_path(name)]
            with archive.open(entry) as stream:
                if np.lib.format.read_magic(stream) != (1, 0):
                    raise ValueError(f"Unsupported NPY version for {name}")
                shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                if dtype.str != info["dtype"] or list(shape) != info["shape"]:
                    raise ValueError(f"NPY shape or dtype does not match manifest for {name}")
                if stream.tell() + math.prod(shape) * dtype.itemsize != entry.file_size:
                    raise ValueError(f"Invalid NPY byte size for {name}")
                if name in selected:
                    stream.seek(0)
                    arrays[name] = np.load(stream, allow_pickle=False)
                else:
                    while stream.read(_CHUNK_BYTES):
                        pass
    return manifest, arrays


def read_result(path):
    """Read the fields used by existing analysis scripts from JSON or ZIP."""
    if str(path).lower().endswith(".zip"):
        manifest, arrays = read_archive(path)
        if "results" not in manifest:
            raise ValueError("Archive has no result fields for this analysis")
        return _resolve_results(manifest["results"], arrays)
    with open(path, encoding="utf-8-sig") as source:
        return json.load(source)


def plot_archive(path):
    from matplotlib import pyplot as plt
    from plotting import (plot_bifurcation_extrema, plot_fourier_transform, plot_hysteresis_loop,
                          plot_mLCE_regression, plot_phase_pair, plot_phase_portrait, plot_v, plot_x, plot_zero_one)

    manifest, arrays = read_archive(path, load_arrays="plots")
    handlers = {
        "displacement": plot_x,
        "velocity": plot_v,
        "phase_portrait": plot_phase_portrait,
        "hysteresis_loop": plot_hysteresis_loop,
        "fourier_spectrum": plot_fourier_transform,
        "bifurcation": plot_bifurcation_extrema,
        "zero_one": plot_zero_one,
        "divergence_regression": plot_mLCE_regression,
        "power_law": plot_mLCE_regression,
        "phase_pair": plot_phase_pair,
        "directional_phase_pair": plot_phase_pair,
    }
    figures = []
    for plot in manifest["plots"]:
        kind = plot["kind"]
        options = {name: plot[name] for name in _LABELS & plot.keys()}
        if kind == "fourier_spectrum":
            options["dt"] = manifest["parameters"]["dt"]
        fig, ax = plt.subplots()
        handlers[kind](*(arrays[name] for name in PLOT_INPUTS[kind]), ax=ax, show=False, **options)
        figures.append(fig)
    return figures


def main():
    from matplotlib import pyplot as plt

    parser = argparse.ArgumentParser(description="Plot a saved experiment archive.")
    parser.add_argument("archive")
    config = parser.parse_args()
    try:
        plot_archive(config.archive)
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        parser.error(str(exc))
    plt.show()


if __name__ == "__main__":
    main()
