import numpy as np
import math
from numba import jit, prange

# Parameters 
NX, NY, NZ = 200, 200, 30
LX = 113200.0
LY = 88600.0
LZ = 5000.0
DX = LX / (NX - 1)
DY = LY / (NY - 1)
DZ = LZ / (NZ - 1)

K_x = 100.0
K_y = 100.0
lamda = 1.0e-6
DT = 10.0
DEPOSITION_RATE = 0.001

TERRAIN_HEIGHT = 500.0
TERRAIN_WIDTH = 10000.0
TERRAIN_CX = LX / 2.0
TERRAIN_CY = LY / 2.0

SCAVENGING_COEFF = 1.0e-4
CLOUDBASE_HEIGHT = 1000.0
THERMAL_MAX = 0.3
THERMAL_HEIGHT = 2000.0

LEAK_X_IDX = 177
LEAK_Y_IDX = 147
LEAK_Z_IDX = max(1, min(NZ-1, int(100.0 / DZ)))

plane = NX * NY

# Helper functions
# Compute wind speed at a given altitude using a logarithmic surface profile.
@jit(nopython=True)
def wind_profile(z, u_surface):
    z0 = 0.1
    z_ref = 10.0
    if z < z_ref:
        z = z_ref
    return u_surface * math.log(z / z0) / math.log(z_ref / z0)

# Return the vertical diffusion coefficient as a function of height.
@jit(nopython=True)
def get_Kz(z):
    K_surface = 20.0
    K_top = 5.0
    return K_surface - (K_surface - K_top) * (z / LZ)

# Estimate terrain elevation as a radial Gaussian hill centered in the domain.
@jit(nopython=True)
def terrain_height(x, y):
    dx = x - TERRAIN_CX
    dy = y - TERRAIN_CY
    r2 = dx*dx + dy*dy
    return TERRAIN_HEIGHT * math.exp(-r2 / (2.0 * TERRAIN_WIDTH * TERRAIN_WIDTH))

# Compute thermal vertical motion based on time of day and height.
@jit(nopython=True)
def thermal_vertical_velocity(z, time_h):
    hour = time_h % 24.0
    factor = math.sin(2.0 * math.pi * (hour - 6.0) / 24.0)
    if z > THERMAL_HEIGHT:
        return 0.0
    height_factor = 1.0 - z / THERMAL_HEIGHT
    return THERMAL_MAX * factor * height_factor

# Core wind field recomputation
# Recompute the 3D wind field across the grid, including terrain and thermal effects.
@jit(nopython=True, parallel=True)
def recompute_wind_fields_numba(C, terrain_mask, U_field, V_field, W_field,
                                terrain_z, step, U_wind, V_wind):
    time_h = step * DT / 3600.0
    
    # Precompute 2D terrain slopes and deceleration factors
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
                z_terrain_2d[j, i] = terrain_height(x, y)
    
    for j in range(1, NY-1):
        for i in range(1, NX-1):
            h_xp = z_terrain_2d[j, i+1]
            h_xm = z_terrain_2d[j, i-1]
            h_yp = z_terrain_2d[j+1, i]
            h_ym = z_terrain_2d[j-1, i]
            dhdx[j, i] = (h_xp - h_xm) / (2.0 * DX)
            dhdy[j, i] = (h_yp - h_ym) / (2.0 * DY)
    
    # Radial distance and factor
    x_vals = np.arange(NX) * DX
    y_vals = np.arange(NY) * DY
    factor_2d = np.ones((NY, NX))
    angle_2d = np.zeros((NY, NX))
    for j in range(NY):
        y = y_vals[j]
        for i in range(NX):
            x = x_vals[i]
            dx_cent = x - TERRAIN_CX
            dy_cent = y - TERRAIN_CY
            r = math.sqrt(dx_cent*dx_cent + dy_cent*dy_cent)
            if r < TERRAIN_WIDTH * 1.5:
                factor_2d[j, i] = max(0.2, r / (TERRAIN_WIDTH * 1.5))
            else:
                factor_2d[j, i] = 1.0
            angle_2d[j, i] = math.atan2(dy_cent, dx_cent)
    
    # Main loop 
    for k in prange(NZ):
        z = k * DZ
        u_free = wind_profile(z, U_wind)
        v_free = V_wind
        if z < 1000.0:
            theta = (z / 1000.0) * 0.5
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            u_rot = u_free * cos_t - V_wind * sin_t
            v_rot = u_free * sin_t + V_wind * cos_t
            u_free = u_rot
            v_free = v_rot
        
        for j in range(NY):
            for i in range(NX):
                idx = i + j*NX + k*plane
                if terrain_mask[idx]:
                    U_field[idx] = 0.0
                    V_field[idx] = 0.0
                    W_field[idx] = 0.0
                    continue
                
                factor = factor_2d[j, i]
                u_terr = u_free * factor
                v_terr = v_free * factor
                
                if r < TERRAIN_WIDTH * 1.5:
                    tangential = u_free * 0.3 * math.sin(angle_2d[j, i]) * (1.0 - factor)
                    v_terr += tangential
                
                if z < z_terrain_2d[j, i] + 500.0:
                    lift_factor = math.exp(-z / 500.0)
                    w_orog = (u_terr * dhdx[j, i] + v_terr * dhdy[j, i]) * lift_factor
                else:
                    w_orog = 0.0
                
                w_thermal = thermal_vertical_velocity(z, time_h)
                
                U_field[idx] = u_terr
                V_field[idx] = v_terr
                W_field[idx] = w_orog + w_thermal

# Core time step
# Perform one time step of plume advection, diffusion, deposition, and decay.
@jit(nopython=True, parallel=True)
def step_simulation_numba(C, C_new, ground_dep, terrain_mask, U_field, V_field, W_field,
                          step, precipitation_rate, source_strength):
    wet_decay_global = precipitation_rate * SCAVENGING_COEFF / 3600.0
    
    # Advection and diffusion
    for k in prange(1, NZ-1):
        z = k * DZ
        Kz = get_Kz(z)
        for j in range(1, NY-1):
            for i in range(1, NX-1):
                idx = i + j*NX + k*plane
                
                # Random noise (LCG RNG)
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
                C_new[idx] = C[idx] + DT * (advec_x + advec_y + advec_z + diff_x + diff_y + diff_z + decay) - wet_removal
                
                if wet_removal > 0.0:
                    ground_idx = i + j*NX
                    ground_dep[ground_idx] += wet_removal * DZ
    
    # Boundary conditions and source 
    for k in range(NZ):
        for j in range(NY):
            C_new[0 + j*NX + k*plane] = 0.0
    
    src_idx = LEAK_X_IDX + LEAK_Y_IDX*NX + LEAK_Z_IDX*plane
    C_new[src_idx] = source_strength
    
    for k in range(NZ):
        for j in range(NY):
            C_new[(NX-1) + j*NX + k*plane] = C_new[(NX-2) + j*NX + k*plane]
    
    for k in range(NZ):
        for i in range(NX):
            C_new[i + 0*NX + k*plane] = C_new[i + 1*NX + k*plane]
            C_new[i + (NY-1)*NX + k*plane] = C_new[i + (NY-2)*NX + k*plane]
    
    for i in range(NX):
        for j in range(NY):
            C_new[i + j*NX + (NZ-1)*plane] = C_new[i + j*NX + (NZ-2)*plane]
    
    # Dry deposition
    for i in range(NX):
        for j in range(NY):
            bot = i + j*NX + 0*plane
            up = i + j*NX + 1*plane
            deposited = C_new[up] * DEPOSITION_RATE
            C_new[bot] = C_new[up] * (1.0 - DEPOSITION_RATE)
            ground_dep[i + j*NX] += deposited * DZ
    
    for idx in range(NX*NY*NZ):
        C[idx] = C_new[idx]
    
    # Terrain blocking
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
        
        # Load terrain
        try:
            with open("terrain.bin", "rb") as f:
                data = np.fromfile(f, dtype=np.float64, count=NX*NY)
                if data.size == NX*NY:
                    self.terrain_z = data
                    print("Loaded real terrain from terrain.bin")
                else:
                    raise ValueError
        except:
            print("Warning: using Gaussian mountain fallback")
            self.terrain_z = None
        
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
                        z_terrain = terrain_height(x, y)
                    self.terrain_mask[idx] = 1 if z < z_terrain else 0
        
        # Initial wind fields
        recompute_wind_fields_numba(self.C, self.terrain_mask, self.U_field, self.V_field, self.W_field,
                                    self.terrain_z, self.step, self.U_wind, self.V_wind)
        
        # Stability print
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
                              self.precipitation_rate, self.source_strength)
        self.step += 1
        # Update wind fields every step
        recompute_wind_fields_numba(self.C, self.terrain_mask, self.U_field, self.V_field, self.W_field,
                                    self.terrain_z, self.step, self.U_wind, self.V_wind)
    
    def set_wind(self, u, v):
        self.U_wind = u
        self.V_wind = v
        print(f"Wind set: U={u:.2f} m/s, V={v:.2f} m/s")
        recompute_wind_fields_numba(self.C, self.terrain_mask, self.U_field, self.V_field, self.W_field,
                                    self.terrain_z, self.step, self.U_wind, self.V_wind)
    
    def set_source_strength(self, strength):
        self.source_strength = strength
        print(f"Source strength = {strength:.2f} Bq/m³")
    
    def set_precipitation(self, rate):
        self.precipitation_rate = rate
        print(f"Precipitation rate = {rate:.2f} mm/h")
    
    def get_state(self, output):
        output[:] = self.C
    
    def get_ground_deposition(self, output):
        output[:] = self.ground_dep
    
    def get_step_count(self):
        return self.step
    
    def finalize_simulation(self):
        pass

_model = PlumeModel()

# Export functions
# Initialize the singleton plume model and prepare the simulation state
def init_simulation():
    _model.init_simulation()

# Advance the singleton plume model by one simulation step
def step_simulation():
    _model.step_simulation()

# Set the current wind components for the plume model
def set_wind(u, v):
    _model.set_wind(u, v)

# Update the source strength used by the plume emission source
def set_source_strength(s):
    _model.set_source_strength(s)

# Set the precipitation rate, which controls wet scavenging in the model
def set_precipitation(r):
    _model.set_precipitation(r)

# Copy the current 3D concentration field into the provided output buffer
def get_state(arr):
    _model.get_state(arr)

# Copy the current ground deposition field into the provided output buffer
def get_ground_deposition(arr):
    _model.get_ground_deposition(arr)

# Return the number of simulation steps that have been executed
def get_step_count():
    return _model.get_step_count()

# Finalize or clean up the simulation
def finalize_simulation():
    _model.finalize_simulation()