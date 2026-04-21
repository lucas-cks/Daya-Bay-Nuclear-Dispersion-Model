// plume_3d_step.c - constant wind, symmetric x/y
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

#define NX 200
#define NY 200
#define NZ 50
#define LX 100000.0
#define LY 100000.0
#define LZ 5000.0
#define DX (LX/(NX-1))
#define DY (LY/(NY-1))
#define DZ (LZ/(NZ-1))

#define K_x 100.0
#define K_y 100.0
#define K_z 10.0
#define lamda 1.0e-6
#define DT 10.0
#define SOURCE_CONC 5.0
#define DEPOSITION_RATE 0.001
#define z_speed 0.0          // no vertical wind

static double *C = NULL;
static double *C_new = NULL;
static int plane = 0;
static int step = 0;

static double U_wind = 5.0;   // x-wind (m/s, eastward)
static double V_wind = 0.0;   // y-wind (m/s, northward)

EXPORT void set_wind(double u, double v) {
    U_wind = u;
    V_wind = v;
    printf("Wind: U = %.2f m/s, V = %.2f m/s\n", U_wind, V_wind);
}

EXPORT void init_simulation(void) {
    if (C) free(C);
    if (C_new) free(C_new);
    size_t total = (size_t)NX * NY * NZ;
    C = malloc(total * sizeof(double));
    C_new = malloc(total * sizeof(double));
    if (!C || !C_new) { fprintf(stderr, "Memory allocation failed\n"); exit(1); }
    for (size_t i = 0; i < total; i++) C[i] = 0.0;
    plane = NX * NY;
    step = 0;

    double cfl_x = fabs(U_wind) * DT / DX;
    double cfl_y = fabs(V_wind) * DT / DY;
    double diff_x = K_x * DT / (DX*DX);
    double diff_y = K_y * DT / (DY*DY);
    double diff_z = K_z * DT / (DZ*DZ);
    printf("Stability\nCFL_x=%.4f CFL_y=%.4f\nDiff_x=%.4f Diff_y=%.4f Diff_z=%.4f\n",
           cfl_x, cfl_y, diff_x, diff_y, diff_z);
    if (cfl_x > 1.0 || cfl_y > 1.0 || diff_x > 0.5 || diff_y > 0.5 || diff_z > 0.5)
        printf("WARNING: Unstable\n");
    else
        printf("Stable\n");
}

EXPORT void step_simulation(void) {
    if (!C || !C_new) return;

    #pragma omp parallel for collapse(2)
    for (int k = 1; k < NZ-1; k++) {
        for (int j = 1; j < NY-1; j++) {
            for (int i = 1; i < NX-1; i++) {
                int idx = i + j*NX + k*plane;
                // X advection (upwind)
                double advec_x = (U_wind >= 0) ? -U_wind * (C[idx] - C[idx-1]) / DX
                                              : -U_wind * (C[idx+1] - C[idx]) / DX;
                // Y advection (upwind)
                double advec_y = (V_wind >= 0) ? -V_wind * (C[idx] - C[idx-NX]) / DY
                                              : -V_wind * (C[idx+NX] - C[idx]) / DY;
                double diff_x = K_x * (C[idx+1] - 2*C[idx] + C[idx-1]) / (DX*DX);
                double diff_y = K_y * (C[idx+NX] - 2*C[idx] + C[idx-NX]) / (DY*DY);
                double diff_z = K_z * (C[idx+plane] - 2*C[idx] + C[idx-plane]) / (DZ*DZ);
                double decay = -lamda * C[idx];
                C_new[idx] = C[idx] + DT * (advec_x + advec_y + diff_x + diff_y + diff_z + decay);
            }
        }
    }

    // BCs (unchanged)
    for (int k = 0; k < NZ; k++)
        for (int j = 0; j < NY; j++)
            C_new[0 + j*NX + k*plane] = 0.0;
    int leak_y_idx = NY / 4;
    int leak_z_idx = (int)(100.0 / DZ);
    if (leak_z_idx < 1) leak_z_idx = 1;
    if (leak_z_idx >= NZ) leak_z_idx = NZ-1;
    C_new[0 + leak_y_idx*NX + leak_z_idx*plane] = SOURCE_CONC;

    for (int k = 0; k < NZ; k++) {
        for (int j = 0; j < NY; j++)
            C_new[(NX-1) + j*NX + k*plane] = C_new[(NX-2) + j*NX + k*plane];
        for (int i = 0; i < NX; i++) {
            C_new[i + 0*NX + k*plane] = C_new[i + 1*NX + k*plane];
            C_new[i + (NY-1)*NX + k*plane] = C_new[i + (NY-2)*NX + k*plane];
        }
    }
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            C_new[i + j*NX + (NZ-1)*plane] = C_new[i + j*NX + (NZ-2)*plane];
            int bot = i + j*NX + 0*plane;
            int up  = i + j*NX + 1*plane;
            C_new[bot] = C_new[up] * (1.0 - DEPOSITION_RATE);
        }
    }

    memcpy(C, C_new, (size_t)NX*NY*NZ * sizeof(double));
    step++;
}

EXPORT void get_state(double *output) {
    if (!C) return;
    size_t total = (size_t)NX * NY * NZ;
    for (size_t i = 0; i < total; i++) output[i] = C[i];
}

EXPORT int get_step_count(void) { return step; }

EXPORT void finalize_simulation(void) {
    if (C) { free(C); C = NULL; }
    if (C_new) { free(C_new); C_new = NULL; }
    step = 0;
}