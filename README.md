# 3D Atmospheric Dispersion Engine: Daya Bay Nuclear Incident Simulator

**Author:** Lucas Kai Sing Ching (Department of Physics, CUHK)

## 1\. Overview

This project provides a sophisticated **3D Eulerian Grid Model** designed to simulate the transport, diffusion, and deposition of radioactive isotopes (e.g., I-131) following a hypothetical leak at the **Daya Bay Nuclear Power Plant**.

The simulation domain covers **113.2 km × 88.6 km**, encompassing the Pearl River Delta and Hong Kong, with a vertical reach of **5 km**. By integrating high-resolution terrain data and planetary boundary layer physics, it evaluates the environmental risk and plume trajectory under diverse meteorological conditions.


## 2\. Physics & Methodology

The engine solves the three-dimensional advection-diffusion equation with source and sink terms:

$$\frac{\partial C}{\partial t} = -\vec{u} \cdot \nabla C + \nabla \cdot (K \nabla C) - \lambda C + S$$

### Key Physical Features:

  * **Ekman Spiral Wind Profile:** Unlike simplistic models, this simulator accounts for the Coriolis-induced rotation of wind vectors with height. The wind speed $u(z)$ follows a logarithmic profile near the surface and rotates to align with the geostrophic wind at the top of the boundary layer.
  * **Terrain Interaction:** High-resolution digital elevation data (USGS DEM) is utilized. A **3D Terrain Mask** identifies grid cells intersected by mountains (e.g., Lantau Peak, Tai Mo Shan), treating them as absorption sinks where pollutants are deposited on the windward slopes.
  * **Deposition Mechanisms:**
      * **Dry Deposition:** Parameterized absorption at the ground level.
      * **Wet Deposition (Rainout):** Scavenging of the plume proportional to user-defined rainfall intensity.

-----

## 3\. Implementation: Two Versions

To balance performance and accessibility, this repository offers two distinct implementations:

### A. HPC Version (C + OpenMP + Python) — *Recommended for Research*

This version is built for speed, capable of handling the **1.2 million grid cells** in real-time.

  * **Backend:** Written in C for raw numerical performance.
  * **Parallelization:** Leverages **OpenMP** for multi-threaded spatial computations (advection/diffusion loops).
  * **Interface:** Python-based GUI communicates with the compiled `.dll`/`.so` shared library via `ctypes`.

### B. Portable Version (Pure Python) — *Recommended for Education*

A fully contained Python implementation designed for ease of deployment.

  * **Core:** Optimized using **NumPy** vectorization to mitigate the overhead of interpreted loops.
  * **Dependency-Free:** Requires no C compiler; runs directly on any standard Python environment.

-----

## 4\. Repository Structure

```text
├── src/
│   ├── Daya_Bay_Simulation.c   # High-performance C backend (OpenMP)
│   ├── simulation_gui.py       # Interactive Dashboard & C-Bridge
│   └── pure_python_model.py    # Standalone Python version
├── data/
│   └── terrain.bin             # Binary Digital Elevation Model (HK/Daya Bay)
├── results/                    # Sample plume trajectory plots
└── lib/
    └── libplume.dll            # Pre-compiled C backend (Windows)
```

-----

## 5\. Usage

### HPC Version (C Backend)

1.  **Compile the shared library:**
    ```bash
    gcc -shared -fopenmp -O3 -o libplume.dll src/Daya_Bay_Simulation.c -lm
    ```
2.  **Run the Dashboard:**
    ```bash
    python src/simulation_gui.py
    ```

### Pure Python Version

```bash
python src/pure_python_model.py
```

-----

## 6\. Technical Highlights

  * **Resolution:** $200 \times 200 \times 30$ grid cells.
  * **Time Step:** $\Delta t = 10s$, optimized to satisfy the **CFL (Courant–Friedrichs–Lewy) condition** for numerical stability.
  * **Visualization:** Interactive sliders for **Rainfall Intensity** ($0$ to $50$ mm/h) and **Source Height**, allowing for real-time sensitivity analysis.

-----

## 7\. License

Distributed under the **MIT License**. This simulation is intended for academic and educational purposes.

## 8\. Contact

**Lucas Kai Sing Ching** Department of Physics, The Chinese University of Hong Kong  
*Project Link:* [https://github.com/lucas-cks/Daya-Bayan-Nuclear-Dispersion](https://www.google.com/search?q=https://github.com/lucas-cks/Daya-Bayan-Nuclear-Dispersion)
