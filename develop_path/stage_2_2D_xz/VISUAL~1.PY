import numpy as np
import matplotlib.pyplot as plt
import os

NX, NZ = 200, 50
LX, LZ = 100000.0, 5000.0
x = np.linspace(0, LX/1000.0, NX)
z = np.linspace(0, LZ/1000.0, NZ)

# Debug: Check file exists
if not os.path.exists('output_xz.bin'):
    print("ERROR: output_xz.bin not found!")
    exit(1)

data = np.fromfile('output_xz.bin', dtype=np.float64)
print(f"DEBUG: Total floats read: {len(data)}, Expected: {37*NX*NZ}")

# output 
n_outputs = len(data) // (NX * NZ)
if len(data) % (NX * NZ) != 0:
    print(f"ERROR: Data size {len(data)} not divisible by (NX * NZ)={NX * NZ}")
    exit(1)

actual_outputs = len(data) // (NX * NZ)
print(f"DEBUG: Actual outputs in file: {actual_outputs}, Expected: {n_outputs}")
data = data.reshape(actual_outputs, NZ, NX)
n_outputs = actual_outputs

C = data[-1, :, :]

plt.figure(figsize=(12, 5))
plt.imshow(C, origin='lower', extent=[0, LX/1000.0, 0, LZ/1000.0],
           aspect='auto', cmap='jet', vmin=0, vmax=1)
plt.colorbar(label='Relative concentration')
plt.xlabel('x (km)')
plt.ylabel('z (km)')
plt.title('2D x-z')
plt.savefig('2d_xz.png', dpi=150)
plt.show()