# plume_model.py
import numpy as np
import math
from numba import jit, prange
import os

# Grid & physical parameters
NX, NY, NZ = 200, 200, 30
LX, LY, LZ = 100000.0, 100000.0, 5000.0
DX = LX / (NX - 1)
DY = LY / (NY - 1)
DZ = LZ / (NZ - 1)
plane = NX * NY

K_x = 100.0
K_y = 100.0
lamda = 1.0e-6          # decay constant (s^-1)
DT = 10.0
DEPOSITION_RATE = 0.001

# Terrain parameters (Gaussian mountain)
TERRAIN_HEIGHT = 500.0
TERRAIN_WIDTH = 10000.0
TERRAIN_CX = LX / 2.0
TERRAIN_CY = LY / 2.0

# Wet deposition
SCAVENGING_COEFF = 1.0e-4      # s^-1 per mm/h
CLOUDBASE_HEIGHT = 1000.0

# Thermal circulation
THERMAL_MAX = 0.3
THERMAL_HEIGHT = 2000.0

# help functions
@jit(nopython=True)
def wind_profile(z, u_surface):
    z0 = 0.1
    z_ref = 10.0
    if z < z_ref:
        z = z_ref
    return u_surface * math.log(z / z0) / math.log(z_ref / z0)

@jit(nopython=True)
def get_Kz(z):
    K_surface = 20.0
    K_top = 5.0
    return K_surface - (K_surface - K_top) * (z / LZ)

@jit(nopython=True)
def terrain_height_gauss(x, y):
    dx = x - TERRAIN_CX
    dy = y - TERRAIN_CY
    r2 = dx*dx + dy*dy
    return TERRAIN_HEIGHT * math.exp(-r2 / (2.0 * TERRAIN_WIDTH * TERRAIN_WIDTH))

@jit(nopython=True)
def thermal_vertical_velocity(z, time_h):
    hour = time_h % 24.0
    factor = math.sin(2.0 * math.pi * (hour - 6.0) / 24.0)
    if z > THERMAL_HEIGHT:
        return 0.0
    height_factor = 1.0 - z / THERMAL_HEIGHT
    return THERMAL_MAX * factor * height_factor

# wind field recomputation with terrain effects and thermal circulation
@jit(nopython=True, parallel=True)
def recompute_wind_fields(U_field, V_field, W_field, terrain_mask, terrain_z,
                          step, U_wind, V_wind):
    time_h = step * DT / 3600.0
    # Precompute terrain slopes (2D)
    dhdx = np.zeros((NY, NX))
    dhdy = np.zeros((NY, NX))
    z_terrain_2d = np.zeros((NY, NX))
    if terrain_z is not None:
        for j in range(NY):
            for i in range(NX):
                z_terrain_2d[j, i] = terrain_z[i + j*NX]
    else:
        for j in range(NY):
            y = j * DY
            for i in range(NX):
                x = i * DX
                z_terrain_2d[j, i] = terrain_height_gauss(x, y)
    # Central differences for slope
    for j in range(1, NY-1):
        for i in range(1, NX-1):
            h_xp = z_terrain_2d[j, i+1]
            h_xm = z_terrain_2d[j, i-1]
            h_yp = z_terrain_2d[j+1, i]
            h_ym = z_terrain_2d[j-1, i]
            dhdx[j, i] = (h_xp - h_xm) / (2.0 * DX)
            dhdy[j, i] = (h_yp - h_ym) / (2.0 * DY)
    # Radial distance factor (for wind deceleration)
    x_vals = np.arange(NX) * DX
    y_vals = np.arange(NY) * DY
    factor_2d = np.ones((NY, NX))
    angle_2d = np.zeros((NY, NX))
    for j in range(NY):
        y = y_vals[j]
        for i in range(NX):
            x = x_vals[i]
            dx = x - TERRAIN_CX
            dy = y - TERRAIN_CY
            r = math.sqrt(dx*dx + dy*dy)
            if r < TERRAIN_WIDTH * 1.5:
                factor_2d[j, i] = max(0.2, r / (TERRAIN_WIDTH * 1.5))
            else:
                factor_2d[j, i] = 1.0
            angle_2d[j, i] = math.atan2(dy, dx)
    # Main loop over k
    for k in prange(NZ):
        z = k * DZ
        u_free = wind_profile(z, U_wind)
        v_free = V_wind
        # Ekman spiral (0-1000 m)
        if z < 1000.0:
            theta = (z / 1000.0) * 0.5
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            u_rot = u_free * cos_t - V_wind * sin_t
            v_rot = u_free * sin_t + V_wind * cos_t
            u_free, v_free = u_rot, v_rot
        for j in range(NY):
            y = y_vals[j]
            for i in range(NX):
                idx = i + j*NX + k*plane
                if terrain_mask[idx]:
                    U_field[idx] = V_field[idx] = W_field[idx] = 0.0
                    continue
                factor = factor_2d[j, i]
                u_terr = u_free * factor
                v_terr = v_free * factor
                # Tangential flow around mountain
                r = math.sqrt((x_vals[i]-TERRAIN_CX)**2 + (y-TERRAIN_CY)**2)
                if r < TERRAIN_WIDTH * 1.5:
                    angle = angle_2d[j, i]
                    tangential = u_free * 0.3 * math.sin(angle) * (1.0 - factor)
                    v_terr += tangential
                # Orographic lifting
                z_terrain = z_terrain_2d[j, i]
                if z < z_terrain + 500.0:
                    lift_factor = math.exp(-z / 500.0)
                    w_orog = (u_terr * dhdx[j, i] + v_terr * dhdy[j, i]) * lift_factor
                else:
                    w_orog = 0.0
                w_thermal = thermal_vertical_velocity(z, time_h)
                U_field[idx] = u_terr
                V_field[idx] = v_terr
                W_field[idx] = w_orog + w_thermal

# time step
@jit(nopython=True, parallel=True)
def step_simulation_numba(C, C_new, ground_dep, terrain_mask, U_field, V_field, W_field,
                          step, precipitation_rate, source_strength,
                          leak_x_idx, leak_y_idx, leak_z_idx):
    wet_decay_global = precipitation_rate * SCAVENGING_COEFF / 3600.0
    # Advection & diffusion
    for k in prange(1, NZ-1):
        z = k * DZ
        Kz = get_Kz(z)
        for j in range(1, NY-1):
            for i in range(1, NX-1):
                idx = i + j*NX + k*plane
                # LCG noise
                seed = (i * 73856093 + j * 19349663 + k * 83492791 + step * 10234567) & 0xffffffff
                seed = (seed * 1103515245 + 12345) & 0xffffffff
                random = (seed & 0x7fffffff) / 0x7fffffff
                noise_factor = 0.05
                u_local = U_field[idx]
                v_local = V_field[idx]
                w_local = W_field[idx]
                random_noise = (random - 0.5) * noise_factor * abs(u_local)
                u_eff = u_local + random_noise
                v_eff = v_local + random_noise
                w_eff = w_local + random_noise
                # X advection
                if u_eff >= 0:
                    advec_x = -u_eff * (C[idx] - C[idx-1]) / DX
                else:
                    advec_x = -u_eff * (C[idx+1] - C[idx]) / DX
                # Y advection
                if v_eff >= 0:
                    advec_y = -v_eff * (C[idx] - C[idx-NX]) / DY
                else:
                    advec_y = -v_eff * (C[idx+NX] - C[idx]) / DY
                # Z advection
                if w_eff >= 0:
                    advec_z = -w_eff * (C[idx] - C[idx-plane]) / DZ
                else:
                    advec_z = -w_eff * (C[idx+plane] - C[idx]) / DZ
                # Diffusion
                diff_x = K_x * (C[idx+1] - 2*C[idx] + C[idx-1]) / (DX*DX)
                diff_y = K_y * (C[idx+NX] - 2*C[idx] + C[idx-NX]) / (DY*DY)
                diff_z = Kz * (C[idx+plane] - 2*C[idx] + C[idx-plane]) / (DZ*DZ)
                decay = -lamda * C[idx]
                wet_decay = wet_decay_global if z < CLOUDBASE_HEIGHT else 0.0
                wet_removal = wet_decay * C[idx] * DT
                C_new[idx] = C[idx] + DT * (advec_x + advec_y + advec_z +
                                            diff_x + diff_y + diff_z + decay) - wet_removal
                if wet_removal > 0.0:
                    ground_idx = i + j*NX
                    ground_dep[ground_idx] += wet_removal * DZ
    # Boundary conditions & source
    for k in range(NZ):
        for j in range(NY):
            C_new[0 + j*NX + k*plane] = 0.0
    src_idx = leak_x_idx + leak_y_idx*NX + leak_z_idx*plane
    C_new[src_idx] = source_strength
    # Outflow x=NX-1 (zero gradient)
    for k in range(NZ):
        for j in range(NY):
            C_new[(NX-1) + j*NX + k*plane] = C_new[(NX-2) + j*NX + k*plane]
    # Y boundaries (zero gradient)
    for k in range(NZ):
        for i in range(NX):
            C_new[i + 0*NX + k*plane] = C_new[i + 1*NX + k*plane]
            C_new[i + (NY-1)*NX + k*plane] = C_new[i + (NY-2)*NX + k*plane]
    # Top boundary (zero gradient)
    for i in range(NX):
        for j in range(NY):
            C_new[i + j*NX + (NZ-1)*plane] = C_new[i + j*NX + (NZ-2)*plane]
    # Ground dry deposition
    for i in range(NX):
        for j in range(NY):
            bot = i + j*NX + 0*plane
            up  = i + j*NX + 1*plane
            deposited = C_new[up] * DEPOSITION_RATE
            C_new[bot] = C_new[up] * (1.0 - DEPOSITION_RATE)
            ground_dep[i + j*NX] += deposited * DZ
    # Copy new -> old
    for idx in range(NX*NY*NZ):
        C[idx] = C_new[idx]
    # Terrain blocking (absorption)
    for idx in range(NX*NY*NZ):
        if terrain_mask[idx]:
            removed = C[idx]
            if removed > 0.0:
                ix = idx % NX
                iy = (idx // NX) % NY
                ground_dep[ix + iy*NX] += removed * DZ
            C[idx] = 0.0
    # Decay of ground deposition
    decay_factor = math.exp(-lamda * DT)
    for idx in range(NX*NY):
        ground_dep[idx] *= decay_factor

# Main model class
class PlumeModel:
    def __init__(self):
        self.C = None
        self.C_new = None
        self.terrain_mask = None
        self.ground_dep = None
        self.U_field = None
        self.V_field = None
        self.W_field = None
        self.terrain_z = None
        self.step = 0
        self.U_wind = 5.0
        self.V_wind = 0.0
        self.source_strength = 5.0
        self.precipitation_rate = 0.0
        self.leak_x_idx = 0        # will be set in init
        self.leak_y_idx = NY // 4
        self.leak_z_idx = int(100.0 / DZ)

    def init_simulation(self):
        total = NX * NY * NZ
        self.C = np.zeros(total, dtype=np.float64)
        self.C_new = np.zeros(total, dtype=np.float64)
        self.terrain_mask = np.zeros(total, dtype=np.int8)
        self.U_field = np.zeros(total, dtype=np.float64)
        self.V_field = np.zeros(total, dtype=np.float64)
        self.W_field = np.zeros(total, dtype=np.float64)
        self.ground_dep = np.zeros(NX*NY, dtype=np.float64)
        self.step = 0

        # Load terrain from file if exists, else Gaussian hill
        if os.path.exists("terrain.bin"):
            data = np.fromfile("terrain.bin", dtype=np.float64, count=NX*NY)
            if data.size == NX*NY:
                self.terrain_z = data
                print("Loaded real terrain from terrain.bin")
            else:
                self.terrain_z = None
        else:
            self.terrain_z = None
            print("Using Gaussian mountain (terrain.bin not found)")

        # Precompute terrain mask
        for k in range(NZ):
            z = k * DZ
            for j in range(NY):
                y = j * DY
                for i in range(NX):
                    x = i * DX
                    idx = i + j*NX + k*plane
                    if self.terrain_z is not None:
                        z_terrain = self.terrain_z[i + j*NX]
                    else:
                        z_terrain = terrain_height_gauss(x, y)
                    self.terrain_mask[idx] = 1 if z < z_terrain else 0

        # Set leak position (source at x=0, y=NY/4, height ~100 m)
        self.leak_x_idx = 0
        self.leak_y_idx = NY // 4
        self.leak_z_idx = max(1, min(NZ-1, int(100.0 / DZ)))

        # Initial wind fields
        recompute_wind_fields(self.U_field, self.V_field, self.W_field,
                              self.terrain_mask, self.terrain_z,
                              self.step, self.U_wind, self.V_wind)

        # Stability check
        max_u = wind_profile(LZ, self.U_wind)
        max_Kz = get_Kz(0.0)
        max_w = 0.25 + THERMAL_MAX
        cfl_x = abs(max_u) * DT / DX
        cfl_y = abs(self.V_wind) * DT / DY
        cfl_z = max_w * DT / DZ
        diff_x = K_x * DT / (DX*DX)
        diff_y = K_y * DT / (DY*DY)
        diff_z = max_Kz * DT / (DZ*DZ)
        print(f"Stability: CFL_x={cfl_x:.4f} CFL_y={cfl_y:.4f} CFL_z={cfl_z:.4f} "
              f"Diff_x={diff_x:.4f} Diff_y={diff_y:.4f} Diff_z={diff_z:.4f}")

    def step_simulation(self):
        step_simulation_numba(self.C, self.C_new, self.ground_dep, self.terrain_mask,
                              self.U_field, self.V_field, self.W_field, self.step,
                              self.precipitation_rate, self.source_strength,
                              self.leak_x_idx, self.leak_y_idx, self.leak_z_idx)
        self.step += 1
        # Update wind fields every step (time‑dependent thermal)
        recompute_wind_fields(self.U_field, self.V_field, self.W_field,
                              self.terrain_mask, self.terrain_z,
                              self.step, self.U_wind, self.V_wind)

    def set_wind(self, u, v):
        self.U_wind = u
        self.V_wind = v
        recompute_wind_fields(self.U_field, self.V_field, self.W_field,
                              self.terrain_mask, self.terrain_z,
                              self.step, self.U_wind, self.V_wind)

    def set_source_strength(self, strength):
        self.source_strength = strength

    def set_precipitation(self, rate):
        self.precipitation_rate = rate

    def get_state(self, output):
        output[:] = self.C

    def get_ground_deposition(self, output):
        output[:] = self.ground_dep

    def get_step_count(self):
        return self.step

    def finalize_simulation(self):
        pass  
