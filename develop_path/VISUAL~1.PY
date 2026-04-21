import numpy as np
import matplotlib.pyplot as plt
import os

NX = 200; LX = 100000.0
x = np.linspace(0, LX, NX)

# Check file exists
if not os.path.exists('output.bin'):
    print("ERROR: output.bin not found!")
    exit(1)

data = np.fromfile('output.bin', dtype=np.float64)
print(f"DEBUG: Total floats read: {len(data)}, Expected: {37*NX}")

# output every 100 steps, total 3600 steps => 37 outputs
n_outputs = 37
if len(data) % NX != 0:
    print(f"ERROR: Data size {len(data)} not divisible by NX={NX}")
    exit(1)

actual_outputs = len(data) // NX
print(f"DEBUG: Actual outputs in file: {actual_outputs}, Expected: {n_outputs}")
data = data.reshape(actual_outputs, NX)
n_outputs = actual_outputs

for i in range(0, n_outputs, 5):
    plt.plot(x/1000, data[i], label=f't={i*100*10/3600:.1f}h')
plt.legend()
plt.xlabel('x (km)')
plt.ylabel('Relative concentration (Bq/m³)')
plt.savefig('concentration_profiles.png', dpi=300)
plt.show()