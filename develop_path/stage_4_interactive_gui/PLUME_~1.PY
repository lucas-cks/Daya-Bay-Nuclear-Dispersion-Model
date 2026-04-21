import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.animation import FuncAnimation
import ctypes
import os

lib_path = "./libplume.dll"
if not os.path.exists(lib_path):
    raise FileNotFoundError(f"{lib_path} not found. Compile C code first.")
lib = ctypes.CDLL(lib_path)

lib.init_simulation.argtypes = []
lib.init_simulation.restype = None
lib.step_simulation.argtypes = []
lib.step_simulation.restype = None
lib.get_state.argtypes = [ctypes.POINTER(ctypes.c_double)]
lib.get_state.restype = None
lib.get_step_count.argtypes = []
lib.get_step_count.restype = ctypes.c_int
lib.finalize_simulation.argtypes = []
lib.finalize_simulation.restype = None
lib.set_wind.argtypes = [ctypes.c_double, ctypes.c_double]
lib.set_wind.restype = None

NX, NY, NZ = 200, 200, 50
LX, LY, LZ = 100000.0, 100000.0, 5000.0
DX, DY, DZ = LX/(NX-1), LY/(NY-1), LZ/(NZ-1)
x_km = np.linspace(0, LX/1000, NX)
y_km = np.linspace(0, LY/1000, NY)

lib.init_simulation()
lib.set_wind(5.0, 0.0)   # initial: eastward 5 m/s

state = np.zeros(NX * NY * NZ, dtype=np.float64)

# Source height index
k_source = int(100.0 / DZ)
if k_source < 0: k_source = 0
if k_source >= NZ: k_source = NZ-1
plane = NX * NY

fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(bottom=0.3)

# Initial image (zeros)
img_data = np.zeros((NY, NX))
im = ax.imshow(img_data, origin='lower', extent=[0, LX/1000, 0, LY/1000],
               cmap='jet', aspect='auto')
plt.colorbar(im, ax=ax, label='Concentration')

ax.set_xlabel('x (km)')
ax.set_ylabel('y (km)')
title_text = ax.set_title('Time = 0.0 h, Max = 0.00')

# Sliders – symmetric intervals
ax_u = plt.axes([0.2, 0.20, 0.6, 0.03])
slider_u = Slider(ax_u, 'U wind (m/s, E->W)', -10.0, 10.0, valinit=5.0, valstep=0.1)
ax_v = plt.axes([0.2, 0.15, 0.6, 0.03])
slider_v = Slider(ax_v, 'V wind (m/s, N->S)', -10.0, 10.0, valinit=0.0, valstep=0.1)

# Buttons
ax_reset = plt.axes([0.7, 0.05, 0.1, 0.05])
btn_reset = Button(ax_reset, 'Reset')
ax_pause = plt.axes([0.82, 0.05, 0.1, 0.05])
btn_pause = Button(ax_pause, 'Pause')

paused = False
steps_per_frame = 5

def update_plot(frame):
    global paused
    if not paused:
        for _ in range(steps_per_frame):
            lib.step_simulation()
        ptr = state.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        lib.get_state(ptr)
        start = k_source * plane
        slice_flat = state[start:start + plane]
        slice_2d = slice_flat.reshape((NY, NX))
        im.set_array(slice_2d)
        im.set_clim(vmin=0, vmax=slice_2d.max() if slice_2d.max() > 0 else 1)
        step = lib.get_step_count()
        time_h = step * 10.0 / 3600.0
        max_conc = slice_2d.max()
        title_text.set_text(f'Time = {time_h:.1f} h, Max = {max_conc:.2e}')
    return im, title_text

def reset_sim(event):
    global paused
    lib.finalize_simulation()
    lib.init_simulation()
    lib.set_wind(slider_u.val, slider_v.val)
    ptr = state.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    lib.get_state(ptr)
    start = k_source * plane
    slice_flat = state[start:start + plane]
    slice_2d = slice_flat.reshape((NY, NX))
    im.set_array(slice_2d)
    step = lib.get_step_count()
    time_h = step * 10.0 / 3600.0
    max_conc = slice_2d.max()
    title_text.set_text(f'Time = {time_h:.1f} h, Max = {max_conc:.2e}')
    if paused:
        paused = False
        btn_pause.label.set_text('Pause')

def toggle_pause(event):
    global paused
    paused = not paused
    btn_pause.label.set_text('Resume' if paused else 'Pause')

def update_wind(val):
    lib.set_wind(slider_u.val, slider_v.val)

slider_u.on_changed(update_wind)
slider_v.on_changed(update_wind)
btn_reset.on_clicked(reset_sim)
btn_pause.on_clicked(toggle_pause)

ani = FuncAnimation(fig, update_plot, interval=33, blit=False)  # ~30 fps
plt.show()

lib.finalize_simulation()