// Daya Bay HK Simulation - C implementation of a 3D atmospheric dispersion model with terrain and wet deposition
// gcc -shared -fopenmp -static -O2 -o libplume.dll Daya_Bay_HK_Simulation.c -lm

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

#define M_PI 3.14159265358979323846
#define NX 200
#define NY 200
#define NZ 30
#define LX 113200.0   // x length
#define LY 88600.0    // y length
#define LZ 5000.0
#define DX (LX/(NX-1))
#define DY (LY/(NY-1))
#define DZ (LZ/(NZ-1))

#define K_x 100.0
#define K_y 100.0
#define lamda 1.0e-6
#define DT 10.0
#define DEPOSITION_RATE 0.001
#define z_speed 0.0          // no vertical wind

static double source_strength = 5.0;         // Bq/m^3 at source

// Terrain parameters (Gaussian mountain)
#define TERRAIN_HEIGHT 500.0
#define TERRAIN_WIDTH 10000.0
#define TERRAIN_CX (LX/2.0)
#define TERRAIN_CY (LY/2.0)

// Wet deposition (rain scavenging)
#define SCAVENGING_COEFF 1.0e-4   // s^-1 per mm/h
static double precipitation_rate = 0.0;   // mm/h

// Thermal circulation parameters (mountain-valley breeze)
#define THERMAL_MAX 0.3           // maximum vertical velocity from thermal effect (m/s)
#define THERMAL_HEIGHT 2000.0     // height up to which thermal effect is active (m)

// Wet deposition vertical profile
#define CLOUDBASE_HEIGHT 1000.0   // m, below which rain scavenging occurs

static double *C = NULL;
static double *C_new = NULL;
static int plane = 0;
static int step = 0;
static char *terrain_mask = NULL;      // 1 if inside mountain, 0 otherwise
static double *ground_dep = NULL;      // accumulated deposition (Bq/m^2)

static double *U_field = NULL;          // x-wind (m/s, eastward)
static double *V_field = NULL;          // y-wind (m/s, northward)
static double *W_field = NULL;          // z-wind (m/s, vertical)

static double U_wind = 5.0;             // surface eastward wind (m/s)
static double V_wind = 0.0;             // surface northward wind (m/s)

// Forward declaration
static void recompute_wind_fields(void);
static double wind_profile(double z, double u_surface);
static double get_Kz(double z);
static double terrain_height(double x, double y);
static double *terrain_z = NULL;


// Wind setter 
EXPORT void set_wind(double u, double v) {
    U_wind = u;
    V_wind = v;
    printf("Wind: U = %.2f m/s, V = %.2f m/s\n", U_wind, V_wind);
    if (U_field && V_field && W_field && terrain_mask) {
        recompute_wind_fields();
        // Dynamic CFL check
        double max_u = wind_profile(LZ, U_wind);
        double max_v = fabs(V_wind);
        double cfl_x = max_u * DT / DX;
        double cfl_y = max_v * DT / DY;
        if (cfl_x > 1.0 || cfl_y > 1.0) {
            printf("WARNING: New wind speeds cause CFL_x=%.3f or CFL_y=%.3f > 1.0. Reduce DT or wind speed.\n", cfl_x, cfl_y);
        } else {
        printf("CFL_x=%.3f, CFL_y=%.3f - stable.\n", cfl_x, cfl_y);
        }
    }
}


// Source strength setter 
EXPORT void set_source_strength(double strength) {
    source_strength = strength;
    printf("Source strength = %.2f Bq/m³\n", source_strength);
}


// Retrieve ground deposition map (size NX*NY, Bq/m^2)
EXPORT void get_ground_deposition(double *output) {
    if (!ground_dep) return;
    size_t total = (size_t)NX * NY;
    for (size_t i = 0; i < total; i++) output[i] = ground_dep[i];
}


// Set precipitation rate (mm/h) for wet deposition
EXPORT void set_precipitation(double rate) {
    precipitation_rate = rate;
    printf("Precipitation rate = %.2f mm/h\n", precipitation_rate);
}


// Vertical diffusion coefficient (linear decrease with height)
static double get_Kz(double z) {
    double K_surface = 20.0;
    double K_top = 5.0;
    return K_surface - (K_surface - K_top) * (z / LZ);
}


// Logarithmic wind profile (reference height 10 m)
static double wind_profile(double z, double u_surface) {
    double z0 = 0.1;
    double kappa = 0.4;
    double z_ref = 10.0;
    if (z < z_ref) z = z_ref;
    return u_surface * log(z / z0) / log(z_ref / z0);
}


// Terrain height (Gaussian mountain)
static double terrain_height(double x, double y) {
    double dx = x - TERRAIN_CX;
    double dy = y - TERRAIN_CY;
    double r2 = dx*dx + dy*dy;
    return TERRAIN_HEIGHT * exp(-r2 / (2.0 * TERRAIN_WIDTH * TERRAIN_WIDTH));
}


// Thermal (mountain-valley breeze) vertical velocity
static double thermal_vertical_velocity(double z, double time_h) {
    // time_h: hours since simulation start
    double hour = fmod(time_h, 24.0);
    double factor = sin(2.0 * M_PI * (hour - 6.0) / 24.0); // +1 at 6h (noon), -1 at 18h
    if (z > THERMAL_HEIGHT) return 0.0;
    double height_factor = 1.0 - z / THERMAL_HEIGHT;
    return THERMAL_MAX * factor * height_factor;
}

// Recompute 3D wind fields based on current U_wind, V_wind and terrain mask
static void recompute_wind_fields(void) {
    double time_h = step * DT / 3600.0;
    if (!U_field || !V_field || !W_field || !terrain_mask) return;
    #pragma omp parallel for collapse(2)
    for (int k = 0; k < NZ; k++) {
        double z = k * DZ;
        double u_free = wind_profile(z, U_wind);
        double v_free = V_wind;

        // Ekman spiral: rotate wind direction with height (0-1000 m)
        if (z < 1000.0) {
            double theta = (z / 1000.0) * 0.5;      // 0 to 0.5 rad (~30 degree)
            double cos_t = cos(theta);
            double sin_t = sin(theta);
            double u_rot = u_free * cos_t - V_wind * sin_t;
            double v_rot = u_free * sin_t + V_wind * cos_t;
            u_free = u_rot;
            v_free = v_rot;
        }

        for (int j = 0; j < NY; j++) {
            double y = j * DY;
            for (int i = 0; i < NX; i++) {
                double x = i * DX;
                int idx = i + j*NX + k*plane;
                if (terrain_mask[idx]) {
                    // Inside mountain: no wind
                    U_field[idx] = 0.0;
                    V_field[idx] = 0.0;
                    W_field[idx] = 0.0;
                } else {
                    // Terrain-induced deceleration (distance based)
                    double dx = x - TERRAIN_CX;
                    double dy = y - TERRAIN_CY;
                    double r = sqrt(dx*dx + dy*dy);
                    double factor = 1.0;
                    if (r < TERRAIN_WIDTH * 1.5) {
                        factor = fmax(0.2, r / (TERRAIN_WIDTH * 1.5));
                    }
                    U_field[idx] = u_free * factor;
                    V_field[idx] = v_free * factor;

                    // Compute terrain slope (dh/dx, dh/dy) using central differences
                    double dh_dx = 0.0, dh_dy = 0.0;
                    if (i > 0 && i < NX-1 && j > 0 && j < NY-1) {
                        double xp = (i+1) * DX, xm = (i-1) * DX;
                        double yp = (j+1) * DY, ym = (j-1) * DY;
                        double h_xp, h_xm, h_yp, h_ym;
                    if (terrain_z) {
                        h_xp = terrain_z[(i+1) + j*NX];
                        h_xm = terrain_z[(i-1) + j*NX];
                        h_yp = terrain_z[i + (j+1)*NX];
                        h_ym = terrain_z[i + (j-1)*NX];
                    } else {
                        h_xp = terrain_height(xp, y);
                        h_xm = terrain_height(xm, y);
                        h_yp = terrain_height(x, yp);
                        h_ym = terrain_height(x, ym);
                    }
                        dh_dx = (h_xp - h_xm) / (2.0 * DX);
                        dh_dy = (h_yp - h_ym) / (2.0 * DY);
}

                    // Orographic lifting (vertical wind)
                    double terrain_z_val;
                    if (terrain_z) {
                        terrain_z_val = terrain_z[i + j*NX];
                    } else {
                        terrain_z_val = terrain_height(x, y);
                }
                    if (z < terrain_z_val + 500.0 && !terrain_mask[idx]) {
                        double lift_factor = exp(-z / 500.0);
                        W_field[idx] = (U_field[idx] * dh_dx + V_field[idx] * dh_dy) * lift_factor;
                    } else {
                        W_field[idx] = 0.0;
                    }

                    // Add thermal (diurnal) circulation
                    double time_h = step * DT / 3600.0;   // current simulation time in hours
                    double w_thermal = thermal_vertical_velocity(z, time_h);
                    W_field[idx] += w_thermal;

                    // Tangential flow (horizontal diversion around mountain)
                    if (r < TERRAIN_WIDTH * 1.5) {
                        double angle = atan2(dy, dx);
                        double tangential = u_free * 0.3 * sin(angle) * (1.0 - factor);
                        V_field[idx] += tangential;
                    }
                }
            }
        }
    }
}


// Initialisation
EXPORT void init_simulation(void) {
    // Free any existing memory
    if (C) free(C);
    if (C_new) free(C_new);
    if (terrain_mask) free(terrain_mask);
    if (ground_dep) free(ground_dep);
    if (U_field) free(U_field);
    if (V_field) free(V_field);
    if (W_field) free(W_field);

    size_t total = (size_t)NX * NY * NZ;
    plane = NX * NY;                // must be set before using in index calculations

    // Allocate concentration arrays
    C = malloc(total * sizeof(double));
    C_new = malloc(total * sizeof(double));
    terrain_mask = malloc(total * sizeof(char));
    U_field = malloc(total * sizeof(double));
    V_field = malloc(total * sizeof(double));
    W_field = malloc(total * sizeof(double));
    ground_dep = malloc(NX * NY * sizeof(double));
    // real terrain height array
    terrain_z = malloc(NX * NY * sizeof(double));
    if (!terrain_z) {
        fprintf(stderr, "Warning: terrain_z allocation failed, using Gaussian mountain.\n");
    // if terrain_z allocation fails, we will fall back to the Gaussian mountain in terrain_height()
    } else {
    FILE *tf = fopen("terrain.bin", "rb");
    if (!tf) {
        fprintf(stderr, "Warning: cannot open terrain.bin, using Gaussian mountain.\n");
        free(terrain_z);
        terrain_z = NULL;
    } else {
        size_t nread = fread(terrain_z, sizeof(double), NX * NY, tf);
        if (nread != (size_t)(NX * NY)) {
            fprintf(stderr, "Error reading terrain.bin, incomplete data.\n");
            free(terrain_z);
            terrain_z = NULL;
        }
        fclose(tf);
        printf("Loaded real terrain from terrain.bin\n");
    }
}

    if (!C || !C_new || !terrain_mask || !U_field || !V_field || !W_field || !ground_dep) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    // Initialise concentration to zero
    for (size_t i = 0; i < total; i++) C[i] = 0.0;
    for (size_t i = 0; i < (size_t)NX*NY; i++) ground_dep[i] = 0.0;
    step = 0;

    // Precompute terrain mask (1 = inside mountain, 0 = outside)
    #pragma omp parallel for collapse(2)
    for (int k = 0; k < NZ; k++) {
    double z = k * DZ;
    for (int j = 0; j < NY; j++) {
        double y = j * DY;
        for (int i = 0; i < NX; i++) {
            double x = i * DX;
            int idx = i + j*NX + k*plane;
            double z_terrain;
            if (terrain_z) {
                int idx2D = i + j*NX;
                z_terrain = terrain_z[idx2D];
            } else {
                z_terrain = terrain_height(x, y);   // fallback
            }
            terrain_mask[idx] = (z < z_terrain) ? 1 : 0;
        }
    }
}

    // Precompute wind fields using the terrain mask and current wind speeds
    recompute_wind_fields();

    // Stability check 
    double max_u = wind_profile(LZ, U_wind);
    double max_Kz = get_Kz(0.0);
    double max_w = 0.25 + THERMAL_MAX;   // include thermal contribution
    double cfl_x = fabs(max_u) * DT / DX;
    double cfl_y = fabs(V_wind) * DT / DY;
    double cfl_z = fabs(max_w) * DT / DZ;
    double diff_x = K_x * DT / (DX*DX);
    double diff_y = K_y * DT / (DY*DY);
    double diff_z = max_Kz * DT / (DZ*DZ);
    printf("Stability\nCFL_x=%.4f CFL_y=%.4f CFL_z=%.4f\nDiff_x=%.4f Diff_y=%.4f Diff_z=%.4f\n",
           cfl_x, cfl_y, cfl_z, diff_x, diff_y, diff_z);
    if (cfl_x > 1.0 || cfl_y > 1.0 || cfl_z > 1.0 || diff_x > 0.5 || diff_y > 0.5 || diff_z > 0.5)
        printf("WARNING: Unstable\n");
    else
        printf("Stable\n");
}

// One time step of the simulation
EXPORT void step_simulation(void) {
    if (!C || !C_new) return;

    // Precompute wet decay factor 
    double wet_decay_global = precipitation_rate * SCAVENGING_COEFF / 3600.0;   // s^-1

    // 1. Advection and diffusion (interior points)
    #pragma omp parallel for collapse(2)
    for (int k = 1; k < NZ-1; k++) {
        double z = k * DZ;
        double Kz = get_Kz(z);
        for (int j = 1; j < NY-1; j++) {
            for (int i = 1; i < NX-1; i++) {
                int idx = i + j*NX + k*plane;

                // LCG RNG 
                unsigned int seed = (unsigned int)((unsigned int)i * 73856093u + (unsigned int)j * 19349663u + (unsigned int)k * 83492791u + (unsigned int)step * 10234567u);
                seed = seed * 1103515245 + 12345;
                double random = (double)(seed & 0x7fffffff) / (double)0x7fffffff; // [0,1)

                double u_local = U_field[idx];
                double v_local = V_field[idx];
                double w_local = W_field[idx];
                double noise_factor = 0.05;  // 5% noise
                double random_noise = (random - 0.5) * noise_factor * fabs(u_local);
                double u_eff = u_local + random_noise;
                double v_eff = v_local + random_noise;
                double w_eff = w_local + random_noise;

                // X advection (upwind)
                double advec_x = (u_eff >= 0) ? -u_eff * (C[idx] - C[idx-1]) / DX
                                             : -u_eff * (C[idx+1] - C[idx]) / DX;
                // Y advection (upwind)
                double advec_y = (v_eff >= 0) ? -v_eff * (C[idx] - C[idx-NX]) / DY
                                             : -v_eff * (C[idx+NX] - C[idx]) / DY;
                // Z advection (upwind)
                double advec_z = (w_eff >= 0) ? -w_eff * (C[idx] - C[idx-plane]) / DZ
                               : -w_eff * (C[idx+plane] - C[idx]) / DZ;

                // Diffusion (central differences)
                double diff_x = K_x * (C[idx+1] - 2*C[idx] + C[idx-1]) / (DX*DX);
                double diff_y = K_y * (C[idx+NX] - 2*C[idx] + C[idx-NX]) / (DY*DY);
                double diff_z = Kz * (C[idx+plane] - 2*C[idx] + C[idx-plane]) / (DZ*DZ);

                double decay = -lamda * C[idx];
                double wet_decay = (z < CLOUDBASE_HEIGHT) ? wet_decay_global : 0.0;
                double wet_removal = wet_decay * C[idx] * DT;
                C_new[idx] = C[idx] + DT * (advec_x + advec_y + advec_z + diff_x + diff_y + diff_z + decay) - wet_removal;

                // Add wet-deposited mass to ground deposition (Bq/m^2)
                if (wet_removal > 0.0) {
                    int ground_idx = i + j*NX;
                    #pragma omp atomic
                    ground_dep[ground_idx] += wet_removal * DZ;
                }
            }
        }
    }

    // 2. Boundary conditions
    // Inflow (x=0): zero everywhere, then add point source
    for (int k = 0; k < NZ; k++)
        for (int j = 0; j < NY; j++)
            C_new[0 + j*NX + k*plane] = 0.0;

    int leak_y_idx = 147;                
    int leak_x_idx = 177;                
    int leak_z_idx = (int)(100.0 / DZ);         // source height ~100 m
    if (leak_z_idx < 1) leak_z_idx = 1;
    if (leak_z_idx >= NZ) leak_z_idx = NZ-1;
    C_new[leak_x_idx + leak_y_idx*NX + leak_z_idx*plane] = source_strength;
    
    // Outflow (x=NX-1) zero gradient
    for (int k = 0; k < NZ; k++)
        for (int j = 0; j < NY; j++)
            C_new[(NX-1) + j*NX + k*plane] = C_new[(NX-2) + j*NX + k*plane];

    // Y-boundaries zero gradient
    for (int k = 0; k < NZ; k++) {
        for (int i = 0; i < NX; i++) {
            C_new[i + 0*NX + k*plane] = C_new[i + 1*NX + k*plane];
            C_new[i + (NY-1)*NX + k*plane] = C_new[i + (NY-2)*NX + k*plane];
        }
    }

    // Top (z=NZ-1) zero gradient
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            C_new[i + j*NX + (NZ-1)*plane] = C_new[i + j*NX + (NZ-2)*plane];
        }
    }

    // Ground (z=0) dry deposition 
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            int bot = i + j*NX + 0*plane;
            int up  = i + j*NX + 1*plane;
            double deposited = C_new[up] * DEPOSITION_RATE;
            C_new[bot] = C_new[up] * (1.0 - DEPOSITION_RATE);
            // Add dry-deposited mass to ground deposition array (Bq/m^2)
            int g_idx = i + j*NX;
            #pragma omp atomic
            ground_dep[g_idx] += deposited * DZ;
        }
    }

    // 3. Copy new state to old
    memcpy(C, C_new, (size_t)NX*NY*NZ * sizeof(double));

    // 4. Apply terrain blocking (absorption) 
    #pragma omp parallel for
    for (size_t i = 0; i < (size_t)NX*NY*NZ; i++) {
        if (terrain_mask[i]) {
            double removed = C[i];
            if (removed > 0.0) {
                int ix = i % NX;
                int iy = (i / NX) % NY;
                int g_idx = ix + iy * NX;
                #pragma omp atomic
                ground_dep[g_idx] += removed * DZ;
            }
            C[i] = 0.0;
        }
    }

    // 5. Decay of ground deposition
    double decay_factor = exp(-lamda * DT);
    #pragma omp parallel for
    for (size_t i = 0; i < (size_t)NX*NY; i++) {
        ground_dep[i] *= decay_factor;
}
    step++;
}

// Retrieve current concentration field
EXPORT void get_state(double *output) {
    if (!C) return;
    size_t total = (size_t)NX * NY * NZ;
    for (size_t i = 0; i < total; i++) output[i] = C[i];
}

// Return number of steps executed
EXPORT int get_step_count(void) { return step; }

// Clean up memory
EXPORT void finalize_simulation(void) {
    if (C) { free(C); C = NULL; }
    if (C_new) { free(C_new); C_new = NULL; }
    if (terrain_mask) { free(terrain_mask); terrain_mask = NULL; }
    if (ground_dep) { free(ground_dep); ground_dep = NULL; }
    if (U_field) { free(U_field); U_field = NULL; }
    if (V_field) { free(V_field); V_field = NULL; }
    if (W_field) { free(W_field); W_field = NULL; }
    if (terrain_z) { free(terrain_z); terrain_z = NULL; }
    step = 0;
}