# Experiment results

Build and run the C++ experiment from `PreisachModel/`. The Radons experiment prints the path to the ZIP file saved under `results/` in its working directory. The archive includes the JSON manifest, the numerical arrays and the plot labels.

From `python/`, use the project Python environment with NumPy, SciPy, Matplotlib and aenum to plot a saved result.

```powershell
python .\experiment_archive.py <path-to-result.zip>
```

To receive results automatically, start the local server from `python/`. It saves validated archives under `results/received/` by default.

```powershell
python .\server.py
```

Run the C++ experiment from `PreisachModel/` with delivery enabled.

```powershell
..\bin\Debug-windows-x86_64\PreisachModel\PreisachModel.exe --send
```

The run without `--send` saves the archive locally. If delivery fails, the executable prints the saved path for offline use. Close the server with Ctrl+C.

For a headless check, set `MPLBACKEND=Agg`. A previous Radons JSON result can be checked against a new archive generated with the same experiment parameters.

```powershell
python .\test_experiment_archive.py <path-to-result.zip> <path-to-previous.json>
```

Existing analysis functions in `main.py` accept either the original JSON file or a ZIP archive path. For example, call `TwoTrajsOnCircle(path)` for the full directional analysis. The server and the archive CLI show the first direction as a phase-portrait preview.

With the saved JSON examples present, run `python .\test_archived_results.py` to check the archive views and plot labels against small excerpts of those results.
