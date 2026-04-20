# Daya Bay Nuclear Accident Plume Simulation
**Atmospheric Transport of Radioactive Materials – A 3D Eulerian Dispersion Model**

**Author:** Ching Kai Sing, Lucas  

[![Top Langs](https://github-readme-stats.vercel.app/api/top-langs/?username=lucas-cks&layout=compact&theme=vision-friendly-dark)](https://github.com/anuraghazra/github-readme-stats)

![C](https://img.shields.io/badge/Language-C-blue?logo=c)
![Python](https://img.shields.io/badge/Language-Python-yellow?logo=python)

![OpenMP](https://img.shields.io/badge/Parallel-OpenMP-green?logo=openmp)
![Numba](https://img.shields.io/badge/JIT-Numba-00A3E0?logo=python)

![NumPy](https://img.shields.io/badge/Library-NumPy-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Library-Matplotlib-ffffff?logo=matplotlib&logoColor=black)
![Cartopy](https://img.shields.io/badge/Library-Cartopy-4CAF50?logo=python)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 1. Overview
This project simulates the atmospheric dispersion of radioactive materials following a hypothetical containment breach at the **Daya Bay Nuclear Power Plant** (Guangdong, China). The model solves the 3D advection–diffusion–decay equation on a high‑resolution Eulerian grid, incorporating complex terrain (Hong Kong region), height‑dependent wind shear, diurnal thermal circulations, and both dry/wet deposition.

Two independent implementations are provided:
- **C + OpenMP** – Compiled into a shared library (`libplume.dll`) for maximum performance.
- **Pure Python + Numba** – Just‑in‑time compiled using `numba` for easy experimentation and portability.

A real‑time interactive GUI (using `matplotlib` and `tkinter`) allows users to adjust wind, source strength, rain rate, and viewing height, and to toggle between concentration and ground deposition maps over a realistic geographic basemap (Cartopy).

[Back to Top](#readme-top)

## 2. Physics Background
The model solves the 3D transport equation:

$$\frac{\partial C}{\partial t} + \nabla \cdot (\mathbf{u} C) = \nabla \cdot (K \nabla C) - \lambda C - \Lambda_{\text{dry}} C - \Lambda_{\text{wet}} C$$

Key phenomena implemented:

- **Advection** – 3D wind field (logarithmic profile + Ekman spiral + orographic lifting + thermal breeze).
- **Diffusion** – Constant horizontal ($K_x, K_y = 100\,\text{m}^2/\text{s}$) and height‑dependent vertical ($K_z(z)$) eddy diffusivity.
- **Decay** – First‑order radioactive decay ($\lambda = 1.0\times10^{-6}\,\text{s}^{-1}$).
- **Dry Deposition** – Constant deposition velocity ($v_d = 0.001\,\text{m/s}$) at ground.
- **Wet Deposition** – Rain scavenging with rate proportional to precipitation ($\Lambda_{\text{wet}} = \text{PR} \cdot 1.0\times10^{-4}\,\text{s}^{-1}$ below cloud base).
- **Terrain** – Real topography (USGS data) or a Gaussian mountain fallback; includes flow blocking, orographic lifting, and tangential diversion.
- **Thermal Circulation** – Diurnal mountain‑valley breeze with maximum vertical velocity $0.3\,\text{m/s}$.
- **Stochastic Turbulence** – Pseudo‑random fluctuations added to advection via LCG.

The numerical scheme is **Forward in Time, Upwind in Space (FTUS)**. Stability is ensured by satisfying the CFL and diffusion criteria ($\Delta t = 10\,\text{s}$).

[Back to Top](#readme-top)

## 3. Implementation Details
### Two Parallel Versions
| Feature | C + OpenMP | Pure Python + Numba |
|---------|------------|---------------------|
| Core loops | OpenMP parallel `#pragma omp` | `@jit(nopython=True, parallel=True)` |
| Wind field recomputation | Every time step | Every time step |
| Random number generator | LCG (inlined) | LCG (inlined) |
| Performance | ~10× faster | ~2–3× slower but portable |
| Compilation | Requires C compiler + DLL creation | No compilation (Numba JIT) |
| File | `Daya_Bay_HK_Simulation.c` | `shelter.py` |

### Numerical Grid
- **Domain:** $113.2\,\text{km} \times 88.6\,\text{km} \times 5\,\text{km}$ (covering Daya Bay to Hong Kong).
- **Resolution:** $\Delta x \approx 566\,\text{m}$, $\Delta y \approx 443\,\text{m}$, $\Delta z \approx 172\,\text{m}$.
- **Grid points:** $200 \times 200 \times 30 = 1.2\times10^6$ cells.
- **Time step:** $\Delta t = 10\,\text{s}$.

[Back to Top](#readme-top)

## 4. Repository Structure
```
daya-bay-plume-simulation/
├── Daya_Bay_HK_Simulation.c    # C implementation (compiles to libplume.dll)
├── shelter.py                  # Pure Python + Numba implementation
├── daya_bay.py                 # Main GUI launcher (imports shelter.py)
├── DAYA_B~1.PY                 # Alternative GUI for C DLL (uses ctypes)
├── terrain.bin                 # Real terrain data (USGS, binary double)
├── terrain_bounds.txt          # Geographic bounds (lon_min, lon_max, lat_min, lat_max, anchor)
├── README.md                   # This file
├── LICENSE                     # MIT License
└── requirements.txt            # Python dependencies
```

[Back to Top](#readme-top)

## 5. Code Structure
### C Version (`Daya_Bay_HK_Simulation.c`)
```text
main() (not present – compiled to DLL)
├── init_simulation()
│   ├── allocate arrays
│   ├── load terrain.bin or generate Gaussian mountain
│   ├── build terrain_mask
│   └── recompute_wind_fields()
├── step_simulation()
│   ├── advection/diffusion (upwind + central)
│   ├── wet deposition (if rain)
│   ├── boundary conditions
│   ├── source injection
│   ├── dry deposition
│   └── terrain blocking (absorption)
├── recompute_wind_fields()
│   ├── logarithmic profile
│   ├── Ekman spiral
│   ├── terrain deceleration & orographic lifting
│   └── thermal circulation
└── exported functions: set_wind, set_source_strength, get_ground_deposition, etc.
```

### Python Version (`shelter.py` + `daya_bay.py`)
```text
PlumeModel class
├── init_simulation()
├── step_simulation() → step_simulation_numba()
├── set_wind() → recompute_wind_fields_numba()
├── get_state() / get_ground_deposition()
└── finalize_simulation()

daya_bay.py (GUI)
├── show_setup_dialog()  (tkinter accident selector)
├── matplotlib Figure with Cartopy basemap
├── sliders for U, V, source strength (log), height, rain
├── buttons: Reset, Pause, Toggle Deposition
└── FuncAnimation drives step_simulation()
```

[Back to Top](#readme-top)

## 6. Usage
### Installation
```bash
git clone https://github.com/lucas-cks/daya-bay-plume-simulation.git
cd daya-bay-plume-simulation
```

### Option A: Pure Python + Numba (Recommended for most users)
No compilation required. Install dependencies:
```bash
pip install -r requirements.txt
```
Then run:
```bash
python daya_bay.py
```

### Option B: C + OpenMP (Windows DLL)
Compile the C code into a shared library:
```bash
gcc -shared -fopenmp -static -O2 -o libplume.dll Daya_Bay_HK_Simulation.c -lm
```
Then run the wrapper:
```bash
python DAYA_B~1.PY
```
(Ensure `libplume.dll` is in the same directory.)

### Requirements
- **Python 3.8+** with:
  - `numpy`, `matplotlib`, `numba`, `tkinter` (built‑in), `ctypes` (built‑in)
  - `cartopy` (optional, for realistic basemap; falls back to simple axes)
- **C compiler** (only for Option B): GCC with OpenMP support.

Install all Python dependencies:
```bash
pip install numpy matplotlib numba cartopy
```

[Back to Top](#readme-top)

## 7. Key Results & Validation
The model reproduces expected plume behaviour under different meteorological conditions:

| Scenario | Wind (U, V) | Rain | Terrain | Observed Effect |
|----------|-------------|------|---------|------------------|
| Westward breeze | (-5, 0) m/s | 0 | Real HK | Plume travels toward Hong Kong, partially blocked by Tai Mo Shan |
| Eastward sea breeze | (3, 0) m/s | 0 | Real | Plume disperses over the South China Sea |
| Heavy rain | (-5, 0) m/s | 20 mm/h | Real | Ground deposition sharply reduced; wet scavenging dominant |
| No terrain | (-5, 0) m/s | 0 | Gaussian | Symmetric Gaussian‑like plume without orographic deflection |

### Example output (Pure Python version)
![Plume concentration](docs/sample_concentration.png)  
*Simulated concentration at 100 m height after 2 hours, U = -5 m/s, real terrain.*

The model has been tested for numerical stability with $\Delta t = 10\,\text{s}$:
- Max CFL_x ≈ 0.09, CFL_y ≈ 0.01, CFL_z ≈ 0.02 (well below 1.0)
- Diffusion numbers < 0.5

[Back to Top](#readme-top)

## 8. License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## 9. References
1. Turner, D. B. (1994). *Workbook of Atmospheric Dispersion Estimates*. CRC Press.
2. Venkatram, A., & Wyngaard, J. C. (1988). *Lectures on Air Pollution Modeling*. AMS.
3. USGS EarthExplorer – Terrain data for N22E113, N22E114 (SRTM 1 arc‑second).
4. Wigner, E. P., & Wilkins, J. E. (1944). *Effect of the Temperature of the Moderator on the Velocity Distribution of Neutrons* – for inspiration on numerical methods.

## 10. Contact
For questions, suggestions, or collaboration, please open an issue on this repository or contact the author directly.

**Ching Kai Sing, Lucas**  
Department of Physics, The Chinese University of Hong Kong  
*Project Link:* [https://github.com/lucas-cks/daya-bay-plume-simulation](https://github.com/lucas-cks/daya-bay-plume-simulation)

[Back to Top](#readme-top)
