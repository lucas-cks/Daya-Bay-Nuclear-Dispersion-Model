# Atmospheric Transport of Radioactive Materials
**A Simulation Study of a Hypothetical Daya Bay Nuclear Incident**

**Author:** Ching Kai Sing, Lucas  

[![Top Langs](https://github-readme-stats.vercel.app/api/top-langs/?username=lucas-cks&layout=compact&theme=vision-friendly-dark)](https://github.com/anuraghazra/github-readme-stats)

![C](https://img.shields.io/badge/Language-C-blue?logo=c)
![Python](https://img.shields.io/badge/Language-Python-yellow?logo=python)

![OpenMP](https://img.shields.io/badge/Parallel-OpenMP-green?logo=openmp)
![Numba](https://img.shields.io/badge/JIT-Numba-00A3E0?logo=python)
![Tkinter](https://img.shields.io/badge/GUI-Tkinter-lightgrey?logo=python) 

![NumPy](https://img.shields.io/badge/Library-NumPy-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Library-Matplotlib-ffffff?logo=matplotlib&logoColor=black)
![Cartopy](https://img.shields.io/badge/Library-Cartopy-4CAF50?logo=python)
![math](https://img.shields.io/badge/Library-math-blue?logo=python)
![os](https://img.shields.io/badge/Library-os-green?logo=python) 

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 1. Overview
This repository implements a high-resolution 3D Eulerian dispersion model to simulate atmospheric dispersion, deposition, and radioactive decay following a hypothetical containment breach at the **Daya Bay Nuclear Power Plant** (Guangdong, China). The model includes terrain effects (Hong Kong region), vertical wind shear, diurnal thermal circulation, dry and wet deposition, and stochastic turbulence.

Two independent implementations are included:
- HPC Solver (C, OpenMP) located in `HPC_Solver_Engine/` – for maximum performance; provided as source and compiled DLL.
- Vectorised Python (NumPy + Numba) located in `vectorised_python/` – portable and easier to modify/experiment with.

A graphical user interface (matplotlib + tkinter) allows interactive control of wind, source strength, rain rate and viewing height, and toggling between concentration and ground-deposition visualizations over a geographic basemap (Cartopy, if available).

[Back to Top](#readme-top)

## 2. Physics Background
The model solves the 3D advection–diffusion–decay equation:

$$
\frac{\partial C}{\partial t} + \nabla \cdot (\mathbf{u} C) = \nabla \cdot (K \nabla C) - \lambda C - \Lambda_{\text{dry}} C - \Lambda_{\text{wet}} C
$$

Key processes:
- Advection — 3D wind field combining a logarithmic profile, Ekman spiral, terrain-induced deceleration and orographic lifting, and an optional diurnal thermal breeze.
- Diffusion — constant horizontal eddy diffusivity (typical values used) and height-dependent vertical diffusivity K_z(z).
- Radioactive decay — first-order decay parameter λ (configurable).
- Dry deposition — surface deposition velocity applied at ground cells.
- Wet deposition — rain scavenging parameterised proportional to precipitation below cloud base.
- Terrain — real topography (USGS-derived terrain grid) or a synthetic Gaussian mountain fallback.
- Thermal circulation — diurnal mountain–valley breeze profile.
- Stochastic turbulence — simple LCG-based perturbations added to advection for subgrid variability.

**Mathematical Implementation**
The simulation utilizes a Finite Difference Method with a Forward in Time, Upwind in Space (FTUS) scheme to maintain numerical stability and prevent unphysical negative concentrations.

Stability Criteria:
The engine strictly adheres to the following conditions to ensure mathematical convergence:
1. CFL Condition: $$\frac{U \Delta t}{\Delta x} \leq 1$$ (Advection stability)
2. Diffusion Criterion: $$\frac{K \Delta t}{\Delta x^2} \leq 0.5$$ (Numerical spreading stability)

[Back to Top](#readme-top)

## 3. Implementation Details
### Two Main Implementations
| Feature | HPC (C + OpenMP) | Vectorised Python (NumPy + Numba) |
|---------|------------------|------------------------------------|
| Source files | `HPC_Solver_Engine/daya_bay_HK_simulation.c` | `vectorised_python/shelter.py` |
| GUI | `HPC_Solver_Engine/daya_bay_gui.py` (wraps DLL) | `vectorised_python/daya_bay.py` (pure Python GUI) |
| Parallelism | OpenMP `#pragma omp` in core loops | `numba.jit(nopython=True, parallel=True)` in heavy loops |
| RNG | Inline LCG in C | Inline LCG in Python (Numba) |
| Performance | Highest (native compiled code) | Portable, easier to read and modify |
| Binary | `HPC_Solver_Engine/libplume.dll` (provided) | No binary required; runtime JIT via Numba |

### Numerical Grid (typical configuration)
- Domain: ~113.2 km (x) × 88.6 km (y) × 5 km (z) (covers Daya Bay → Hong Kong).
- Resolution: Δx ≈ 566 m, Δy ≈ 443 m, Δz ≈ 172 m (example grid).
- Grid points: ~200 × 200 × 30 = 1.2×10^6 cells (configurable).
- Time step: Δt = 10 s (used to satisfy stability criteria).

[Back to Top](#readme-top)

## 4. Repository Structure
```
A-Simulation-Study-of-a-Hypothetical-Daya-Bay-Nuclear-Incident/
├── HPC_Solver_Engine/
│   ├── daya_bay_HK_simulation.c   # C solver source (OpenMP)
│   ├── daya_bay_gui.py            # Python GUI wrapper for compiled DLL
│   ├── libplume.dll               # Compiled DLL (Windows) — included
│   ├── terrain.bin                # Terrain binary for solver (USGS grid)
│   └── terrain_bounds.txt         # Geographic bounds for terrain grid
├── vectorised_python/
│   ├── shelter.py                 # Vectorised model (NumPy + Numba)
│   ├── daya_bay.py                # Pure Python GUI (matplotlib + tkinter)
│   ├── terrain.bin                # Copy of terrain data used by Python version
│   └── terrain_bounds.txt
├── develop_path/                  # Stages of development
├── LICENSE                        # MIT License
└── README.md                     
```

[Back to Top](#readme-top)

## 5. Code Structure (high-level)
### HPC_Solver_Engine (C)
- daya_bay_HK_simulation.c:
  - init_simulation(): allocate arrays, load terrain or build Gaussian mountain, build terrain mask, initialize fields.
  - recompute_wind_fields(): logarithmic profile, Ekman spiral, terrain deceleration, orographic lifting, thermal breeze.
  - step_simulation(): advection/diffusion update (FTUS/upwind), wet/dry deposition, boundary handling, source injection, terrain absorption.
  - Exported functions for Python wrapper: set_wind, set_source_strength, get_ground_deposition, step, etc.
- daya_bay_gui.py:
  - ctypes wrapper to call functions in `libplume.dll`.
  - GUI with sliders for wind, source, height, rain; visualization using matplotlib.

### Vectorised Python
- shelter.py:
  - PlumeModel class encapsulating simulation state and core stepping routines (Numba-accelerated functions).
  - Methods: init_simulation(), step_simulation(), set_wind(), get_ground_deposition(), finalize().
- daya_bay.py:
  - GUI launcher (tkinter dialogs + matplotlib animation).
  - Interactive controls: U, V, source strength, height, rain; Reset/Pause/Toggle deposition.

[Back to Top](#readme-top)

## 6. Usage
### Clone the repository
```bash
git clone https://github.com/lucas-cks/A-Simulation-Study-of-a-Hypothetical-Daya-Bay-Nuclear-Incident.git
cd A-Simulation-Study-of-a-Hypothetical-Daya-Bay-Nuclear-Incident
```
> **Note on Cartopy:** If you encounter issues installing Cartopy on Windows, we recommend using the `Conda` package manager: 
> `conda install -c conda-forge cartopy`

### Option A — Vectorised Python (recommended for ease of use)
1. Create a Python 3.8+ environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate      # or venv\Scripts\activate on Windows
pip install -r vectorised_python/requirements.txt  # or install manually
```
If no requirements file exists, install:
```bash
pip install numpy matplotlib numba cartopy
```
2. Run the GUI:
```bash
python vectorised_python/daya_bay.py
```
This runs the pure‑Python GUI and model (Numba JIT acceleration).

### Option B — HPC Solver (C + OpenMP)
The repository includes a precompiled `libplume.dll` in `HPC_Solver_Engine/` for Windows. To rebuild from source:

Linux / macOS (shared object):
```bash
gcc -shared -fopenmp -fPIC -O3 -o libplume.so HPC_Solver_Engine/daya_bay_HK_simulation.c -lm
```

Windows (MinGW / GCC):
```bash
gcc -shared -fopenmp -O2 -o libplume.dll HPC_Solver_Engine/daya_bay_HK_simulation.c -lm
```

Then run the C GUI wrapper:
```bash
python HPC_Solver_Engine/daya_bay_gui.py
```
Ensure the shared library (`libplume.dll` or `libplume.so`) is in the same directory as the GUI script, or update the ctypes loader path.

### Requirements summary
- Python 3.8+:
  - numpy, matplotlib, numba, cartopy (optional for basemap), tkinter (built-in), ctypes (built-in)
- C compiler with OpenMP support (only needed to compile the C solver)
  - gcc, clang (with OpenMP) or MSVC-compatible toolchain

[Back to Top](#readme-top)

## 7. Example Scenarios & Validation
The model reproduces expected plume behaviour in representative scenarios:

| Scenario | Wind (U, V) | Rain | Terrain | Observed Effect |
|----------|-------------|------|---------|-----------------|
| Westward breeze | (-5, 0) m/s | 0 | Real HK | Plume advects toward Hong Kong; partial blocking by Tai Mo Shan |
| Eastward sea breeze | (3, 0) m/s | 0 | Real | Plume disperses over the South China Sea |
| Heavy rain | (-5, 0) m/s | 20 mm/h | Real | Wet scavenging reduces airborne concentration; deposition redistributed |
| No terrain | (-5, 0) m/s | 0 | Gaussian mountain | Symmetric Gaussian-like plume, no orographic deflection |

Numerical stability checks (example configuration):
- Δt = 10 s with CFL_x ≈ 0.09, CFL_y ≈ 0.01, CFL_z ≈ 0.02 (safely < 1.0)
- Diffusion stability numbers < 0.5

[Back to Top](#readme-top)

## 8. Notes on Data and Terrain
- Terrain provided in `HPC_Solver_Engine/terrain.bin` and `vectorised_python/terrain.bin` (binary double grid) with geographic bounds in `terrain_bounds.txt`. These files were generated from SRTM/USGS sources for the target region.
- If terrain files are missing, the code can generate a synthetic Gaussian mountain for testing.

[Back to Top](#readme-top)

## 9. Extending & Modifying
- To test different atmospheric physics, modify wind-field generation in:
  - C: `HPC_Solver_Engine/daya_bay_HK_simulation.c` → recompute_wind_fields()
  - Python: `vectorised_python/shelter.py` → recompute_wind_fields_numba() (or set_wind())
- To change grid resolution or domain, adjust initialization parameters in the respective init_simulation() functions.
- Add modules to export 2D/3D output (NetCDF/GRIB) or integrate with GIS tools for post-processing.

[Back to Top](#readme-top)

## 10. License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 11. References
1. USGS EarthExplorer – Terrain data (SRTM 1 arc-second) for the study region.

## 12. Contact
For questions, suggestions, or collaboration, please open an issue on this repository or contact the author.

**Ching Kai Sing, Lucas**  
Department of Physics, The Chinese University of Hong Kong  
Project link: https://github.com/lucas-cks/A-Simulation-Study-of-a-Hypothetical-Daya-Bay-Nuclear-Incident

[Back to Top](#readme-top)
```
