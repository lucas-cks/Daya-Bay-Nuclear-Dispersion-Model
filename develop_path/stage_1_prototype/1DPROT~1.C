//1D Prototype
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//Grid parameters
#define NX 200
#define LX 100000.0   // 100 km
#define DX (LX/(NX-1))

//Physical parameters
#define U_speed 5.0     //Wind speed m/s
#define K_const 100.0   //Diffusion Constant m^2/s
#define lamda 1.0e-6    //Decay Constant s^-1

//Time parameters
#define DT 10.0
#define NT 3600 //total setps (10 hours)

#define SOURCE_CONC 1.0 //Source concentration at the inflow boundary

int main() {
    
    double cfl = U_speed * DT / DX;
    double diff_number = K_const * DT / (DX * DX);

    printf("CFL Number: %f (Must be < 1.0)\n", cfl);
    printf("Diffusion Number: %f (Must be < 0.5)\n", diff_number);

    if (cfl > 1.0 || diff_number > 0.5) {
        printf("Warning: Stability condition violated! Adjust DT or DX.\n");
    return 1; 
}
    double *C = malloc(NX * sizeof(double));
    double *C_new = malloc(NX * sizeof(double));
    
    // initialise 
    for (int i = 0; i < NX; i++) C[i] = 0.0;
    
    
    // output
    FILE *f = fopen("output.bin", "wb");
    
    // time loop
    for (int n = 0; n <= NT; n++) {
        if (n % 100 == 0) {
            fwrite(C, sizeof(double), NX, f);
        }
        
        for (int i = 1; i < NX-1; i++) {
            double advec = -U_speed * (C[i] - C[i-1]) / DX;
            double diff = K_const * (C[i+1] - 2*C[i] + C[i-1]) / (DX*DX);
            double decay = -lamda * C[i];
            C_new[i] = C[i] + DT * (advec + diff + decay);
        }
        
        // boundary condition
        C_new[0] = SOURCE_CONC; // Inflow boundary
        C_new[NX-1] = C[NX-1]; // Outflow boundary (zero gradient)
        
        // update
        for (int i = 0; i < NX; i++) C[i] = C_new[i];
    }
    
    fclose(f);
    free(C); free(C_new);
    return 0;
}