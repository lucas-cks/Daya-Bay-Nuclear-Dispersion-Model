import numpy as np

# Grid parameters
NX = 200
LX = 100000.0      # 100 km
DX = LX / (NX - 1)

# Physical parameters
U_speed = 5.0      # Wind speed m/s
K_const = 100.0    # Diffusion constant m^2/s
lamda = 1.0e-6     # Decay constant s^-1

# Time parameters
DT = 10.0
NT = 3600          # total steps (10 hours)

SOURCE_CONC = 1.0  # Source concentration at inflow boundary

# Stability checks
cfl = U_speed * DT / DX
diff_number = K_const * DT / (DX * DX)

print(f"CFL Number: {cfl:.6f} (Must be < 1.0)")
print(f"Diffusion Number: {diff_number:.6f} (Must be < 0.5)")

if cfl > 1.0 or diff_number > 0.5:
    print("Warning: Stability condition violated! Adjust DT or DX.")
    exit(1)

# Initialize concentration arrays
C = np.zeros(NX)
C_new = np.zeros(NX)

# Open binary output file
with open("output.bin", "wb") as f:
    # Time loop
    for n in range(NT + 1):
        if n % 100 == 0:
            # Write current concentration array as binary doubles
            C.astype(np.float64).tofile(f)

        # Update interior points
        for i in range(1, NX - 1):
            advec = -U_speed * (C[i] - C[i-1]) / DX
            diff = K_const * (C[i+1] - 2*C[i] + C[i-1]) / (DX * DX)
            decay = -lamda * C[i]
            C_new[i] = C[i] + DT * (advec + diff + decay)

        # Boundary conditions
        C_new[0] = SOURCE_CONC          # Inflow boundary
        C_new[NX-1] = C[NX-1]          # Outflow boundary (zero gradient)

        # Update solution for next time step
        C[:] = C_new[:]

print("Simulation finished. Output written to output.bin")