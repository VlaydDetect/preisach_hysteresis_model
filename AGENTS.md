# Repository Guidelines

## Project Structure & Module Organization

- `PreisachModel/source/` contains the C++ experiment driver (`main.cpp`) and utilities for files, networking, profiling, and numerical work.
- `PreisachModel/vendor/MUTCpp/include/MUTCpp/` contains the hysteresis models and ODE algorithms. Other vendored libraries include Eigen, hep-mc, and Tracy; avoid unrelated dependency edits.
- `python/` contains analysis scripts, plotting helpers, interactive demos, and notebooks. Reuse `plotting.py` and `utils.py`.
- `PreisachModel/` holds simulation data; `python/article_plots/`, `python/pl/`, and `ShuttlePoints/` hold research figures. Build outputs go to `bin/` and `intermediate/`.

## Build, Test, and Development Commands

Use Visual Studio 2022 with C++ desktop tools and C++23 support. From the repository root in a Developer PowerShell:

```powershell
.\premake5.exe vs2022
msbuild .\PreisachModel.sln /m /p:Configuration=Debug /p:Platform=x64
```

The first command generates the solution; the second builds it. Other configurations are `ReleaseWithInfo` and `Release`. Edit Premake files rather than generated solution/project files.

- From `PreisachModel/`, run `..\bin\Debug-windows-x86_64\PreisachModel\PreisachModel.exe`. Select the experiment in `source/main.cpp`; there is no experiment CLI.
- From `python/`, run `python test_cone.py` for the cone visualization or `python preisach_density_explorer.py` for the density GUI. Use a virtual environment with the selected script's dependencies; the cone demo needs NumPy, SciPy, and Matplotlib.

Keep these working directories: scripts use relative data paths. Experiments can overwrite output files; preserve needed results before rerunning.

## Coding Style & Naming Conventions

Match surrounding code: four-space indentation, C++ braces on separate lines, PascalCase types/methods, and `m_` members. Python generally uses snake_case functions and four spaces; preserve existing public names. No repository-wide formatter or linter is configured.

## Testing Guidelines

No project-wide test runner or coverage threshold is configured. Existing `python/test*.py` files are exploratory scripts. For numerical changes, add deterministic assertion-based checks named `test_<behavior>.py`, specify tolerances, and run them directly with Python. Rebuild affected C++ configurations and compare relevant trajectories or hysteresis plots; record parameters and commands.

## Commit & Pull Request Guidelines

History uses short imperative subjects such as `Fix shuttle point algorithm` and `Add Fourier spectrum implementation`; follow that style. Keep commits focused. PRs should describe the numerical or behavioral change, link relevant issues, list validation commands/results, and attach before/after plots when outputs change. Exclude incidental binaries, IDE files, and generated research output.
