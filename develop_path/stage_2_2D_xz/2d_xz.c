//2D x-z
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

//Grid parameters
#define NX 200
#define NZ 50
#define LX 100000.0   // 100 km
#define LZ 5000.0     // 5 km
#define DX (LX/(NX-1))
#define DZ (LZ/(NZ-1))

//Physical parameters
#define z_speed 0.01     //Vertical velocity m/s
#define K_x 100.0       //Horizontal Diffusion Constant m^2/s
#define K_z 10.0        //Vertical Diffusion Constant m^2/s
#define lamda 1.0e-6    //Decay Constant s^-1
#define USTAR 0.3       // Friction velocity m/s
#define Z0 0.1          // Roughness length m
#define KAPPA 0.4       // von Kármán constant

//Time parameters
#define DT 10.0
#define NT 3600 //total setps (10 hours)

#define SOURCE_CONC 5.0 //Source concentration at the inflow boundary
#define DEPOSITION_RATE 0.05 // Deposition rate at the ground

double wind_profile(double z) {
    return (USTAR / KAPPA) * log((z + Z0) / Z0);
}

int main() {
    
    double *C = malloc(NX * NZ * sizeof(double));
    double *C_new = malloc(NX * NZ * sizeof(double));
    
    // initialise 
    for (int i = 0; i < NX * NZ; i++) C[i] = 0.0;
    
    // stability check
    double max_u = wind_profile(LZ);  
    double cfl_x = max_u * DT / DX;
    double diff_stable_x = K_x * DT / (DX*DX);
    double diff_stable_z = K_z * DT / (DZ*DZ);
    printf("Stability\n");
    printf("CFL_x = %.4f (<=1)\n", cfl_x);
    printf("Diff_x = %.4f (<=0.5)\n", diff_stable_x);
    printf("Diff_z = %.4f (<=0.5)\n", diff_stable_z);
    if (cfl_x > 1.0 || diff_stable_x > 0.5 || diff_stable_z > 0.5)
        printf("WARNING: Unstable, Reduce DT.\n");
    else
        printf("Stable.\n");

    // output
    FILE *f = fopen("output_xz.bin", "wb");
    
    // time loop
    for (int n = 0; n <= NT; n++) {
        if (n % 100 == 0) {
            fwrite(C, sizeof(double), NX * NZ, f);
        }
     
    for (int k = 1; k < NZ-1; k++) {
            double z = k * DZ;
            double u_z = wind_profile(z);

        for (int i = 1; i < NX-1; i++) {
            
            int idx = i + k * NX;
            
            double advec_x = -u_z * (C[idx] - C[idx-1]) / DX;
            double diff_x = K_x * (C[idx+1] - 2*C[idx] + C[idx-1]) / (DX*DX);

            double advec_z = -z_speed * (C[idx] - C[idx-NX]) / DZ; 
            double diff_z = K_z * (C[idx+NX] - 2*C[idx] + C[idx-NX]) / (DZ*DZ);

            double decay = -lamda * C[idx];

            C_new[idx] = C[idx] + DT * (advec_x + diff_x + advec_z + diff_z + decay);
        }
     
    }
        // boundary condition

        // 1. Inflow at x=0 
        for (int k = 0; k < NZ; k++) C_new[0 + k * NX] = 0.0;
    
        int leak_z_idx = 20; // Leak at z=2000m (k=20)
        C_new[0 + leak_z_idx * NX] = SOURCE_CONC;

        // 2. Outflow at x=LX (Zero-gradient)
        for (int k = 0; k < NZ; k++) C_new[(NX-1) + k * NX] = C_new[(NX-2) + k * NX]; 

        // 3. Top Boundary (Zero-gradient ceiling)
        for (int i = 0; i < NX; i++) C_new[i + (NZ-1) * NX] = C_new[i + (NZ-2) * NX];

        // 4. Ground: Dry Deposition 
        for (int i = 0; i < NX; i++) {
            int idx_bottom = i + 0 * NX;
            int idx_above  = i + 1 * NX;
            // C_new[idx_bottom] = C_new[idx_above] * (1.0 - DEPOSITION_RATE);
            C_new[idx_bottom] = C_new[idx_above] * (1.0 - DEPOSITION_RATE);
        }

        // update
        memcpy(C, C_new, NX * NZ * sizeof(double));
    }
    
    fclose(f);
    free(C); free(C_new);
    return 0;
}
