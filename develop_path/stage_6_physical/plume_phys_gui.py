import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.animation import FuncAnimation
import matplotlib.colors as colors
import ctypes
import os
import tkinter as tk
from tkinter import ttk

# Increase default font sizes 
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# load dll
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
lib.set_source_strength.argtypes = [ctypes.c_double]
lib.set_source_strength.restype = None
lib.get_ground_deposition.argtypes = [ctypes.POINTER(ctypes.c_double)]
lib.get_ground_deposition.restype = None
lib.set_precipitation.argtypes = [ctypes.c_double]
lib.set_precipitation.restype = None

# Accident database
accidents = {
    "Chernobyl (1986, INES 7)": {
        "source_Bq": 5e6,
        "desc": "Major accident – large radioactive release",
        "ines": 7
    },
    "Fukushima (2011, INES 7)": {
        "source_Bq": 1e6,
        "desc": "Earthquake/tsunami led to meltdown",
        "ines": 7
    },
    "Kyshtym (1957, INES 6)": {
        "source_Bq": 3e5,
        "desc": "Waste tank explosion (Soviet Union)",
        "ines": 6
    },
    "Three Mile Island (1979, INES 5)": {
        "source_Bq": 5e4,
        "desc": "Partial core meltdown, limited release",
        "ines": 5
    },
    "Windscale Fire (1957, INES 5)": {
        "source_Bq": 8e4,
        "desc": "Reactor fire (UK)",
        "ines": 5
    },
    "Tokaimura JCO (1999, INES 4)": {
        "source_Bq": 5e3,
        "desc": "Criticality accident, worker fatalities",
        "ines": 4
    },
    "Custom (user defined)": {
        "source_Bq": 5.0,
        "desc": "Set your own source strength",
        "ines": None
    }
}

# setup window
def show_setup_dialog():
    root = tk.Tk()
    root.title("Radioactive Plume Simulation – Select Accident Level")
    root.geometry("550x450")
    root.resizable(False, False)

    label = tk.Label(root, text="Choose a nuclear accident level (INES) or define custom source strength",
                     font=("Arial", 12))
    label.pack(pady=10)

    selected = tk.StringVar()
    selected.set("Chernobyl (1986, INES 7)")
    combo = ttk.Combobox(root, textvariable=selected, values=list(accidents.keys()),
                         state="readonly", width=45)
    combo.pack(pady=5)

    desc_label = tk.Label(root, text=accidents[selected.get()]["desc"], wraplength=500, justify="left")
    desc_label.pack(pady=5)

    custom_frame = tk.Frame(root)
    custom_lbl = tk.Label(custom_frame, text="Custom source strength (Bq/m³):")
    custom_lbl.pack(side=tk.LEFT, padx=5)
    custom_entry = tk.Entry(custom_frame, width=15)
    custom_entry.insert(0, "5.0")
    custom_entry.pack(side=tk.LEFT)

    def on_accident_change(*args):
        desc_label.config(text=accidents[selected.get()]["desc"])
        if selected.get() == "Custom (user defined)":
            custom_frame.pack(pady=10)
        else:
            custom_frame.pack_forget()

    selected.trace_add("write", on_accident_change)
    on_accident_change()

    wind_frame = tk.Frame(root)
    tk.Label(wind_frame, text="Initial wind U (m/s, eastward):").pack(side=tk.LEFT, padx=5)
    u_entry = tk.Entry(wind_frame, width=8)
    u_entry.insert(0, "5.0")
    u_entry.pack(side=tk.LEFT)
    tk.Label(wind_frame, text="Initial wind V (m/s, northward):").pack(side=tk.LEFT, padx=5)
    v_entry = tk.Entry(wind_frame, width=8)
    v_entry.insert(0, "0.0")
    v_entry.pack(side=tk.LEFT)
    wind_frame.pack(pady=10)

    result = {"source": None, "U": None, "V": None}

    def on_ok():
        if selected.get() == "Custom (user defined)":
            try:
                src = float(custom_entry.get())
            except:
                src = 5.0
        else:
            src = accidents[selected.get()]["source_Bq"]
        try:
            u_val = float(u_entry.get())
            v_val = float(v_entry.get())
        except:
            u_val = 5.0
            v_val = 0.0
        result["source"] = src
        result["U"] = u_val
        result["V"] = v_val
        root.destroy()

    btn_ok = tk.Button(root, text="Start Simulation", command=on_ok, bg="green", fg="white", font=("Arial", 12))
    btn_ok.pack(pady=20)

    root.mainloop()
    return result["source"], result["U"], result["V"]

source_strength, U_init, V_init = show_setup_dialog()
print(f"Starting simulation: source strength = {source_strength:.2e} Bq/m³, U = {U_init:.1f} m/s, V = {V_init:.1f} m/s")

# Grid parameters
NX, NY, NZ = 200, 200, 30
LX, LY, LZ = 100000.0, 100000.0, 5000.0
DX, DY, DZ = LX/(NX-1), LY/(NY-1), LZ/(NZ-1)
x_km = np.linspace(0, LX/1000, NX)
y_km = np.linspace(0, LY/1000, NY)
z_m = np.linspace(0, LZ, NZ)

# Terrain visualisation parameters
TERRAIN_HEIGHT_M = 500.0
TERRAIN_WIDTH_M = 10000.0
CENTER_X_KM = 50.0
CENTER_Y_KM = 50.0

lib.init_simulation()
lib.set_source_strength(source_strength)
lib.set_wind(U_init, V_init)
lib.set_precipitation(0.0)   # start with no rain

state = np.zeros(NX * NY * NZ, dtype=np.float64)
ground_dep_state = np.zeros(NX * NY, dtype=np.float64)   # for deposition map
plane = NX * NY

initial_height = 100.0

# Figure with square aspect
fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(bottom=0.35)   

img_data = np.zeros((NY, NX))
norm = colors.LogNorm(vmin=1e-3, vmax=1e7)
im = ax.imshow(img_data, origin='lower', extent=[0, LX/1000, 0, LY/1000],
               cmap='jet', aspect='equal', norm=norm)

def terrain_height_km(x_km, y_km):
    dx = (x_km - CENTER_X_KM) * 1000.0
    dy = (y_km - CENTER_Y_KM) * 1000.0
    r2 = dx*dx + dy*dy
    return (TERRAIN_HEIGHT_M / 1000.0) * np.exp(-r2 / (2.0 * TERRAIN_WIDTH_M * TERRAIN_WIDTH_M))

X, Y = np.meshgrid(x_km, y_km)
Z_terrain = terrain_height_km(X, Y)

# Colorbar
cb = plt.colorbar(im, ax=ax, label='Concentration (Bq/m³)')
cb.ax.tick_params(labelsize=10)
ticks = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
cb.set_ticks(ticks)
cb.set_ticklabels([f'$10^{{{int(np.log10(t))}}}$' for t in ticks])

# INES level markers 
ines_levels = [
    (1e6, 'INES 7'),
    (3e5, 'INES 6'),
    (5e4, 'INES 5'),
    (5e3, 'INES 4'),
]
for threshold, label in ines_levels:
    pos = (np.log10(threshold) - np.log10(norm.vmin)) / (np.log10(norm.vmax) - np.log10(norm.vmin))
    if 0 < pos < 1:
        cb.ax.axhline(y=pos, color='white', linestyle='--', linewidth=1, alpha=0.7)
        cb.ax.text(0.5, pos, label, transform=cb.ax.transAxes,
                   ha='center', va='bottom', color='white', fontsize=8,
                   bbox=dict(facecolor='black', alpha=0.5, pad=1))

# Terrain contours
contour_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
ax.contour(X, Y, Z_terrain, levels=contour_levels, colors='white', linewidths=1, alpha=0.8)
ax.contourf(X, Y, Z_terrain, levels=[0, 0.5], colors='gray', alpha=0.3)

ax.set_xlabel('x (km)')
ax.set_ylabel('y (km)')
title_text = ax.set_title('Time = 0.0 h, Height = 0 m, Max = 0.00 Bq/m³')

# Sliders
# Adjust vertical positions to make room for rain slider (bottom=0.35 -> bottom=0.40)
plt.subplots_adjust(bottom=0.40)

ax_u = plt.axes([0.2, 0.35, 0.6, 0.03])
slider_u = Slider(ax_u, 'Surface U wind (m/s, W->E)', -10.0, 10.0, valinit=U_init, valstep=0.1)
ax_v = plt.axes([0.2, 0.30, 0.6, 0.03])
slider_v = Slider(ax_v, 'Surface V wind (m/s, S->N)', -10.0, 10.0, valinit=V_init, valstep=0.1)

# Source strength slider (logarithmic)
ax_source = plt.axes([0.2, 0.25, 0.6, 0.03])
slider_source_log = Slider(ax_source, 'Source strength (log10 Bq/m³)', 0, 7, valinit=np.log10(source_strength), valstep=0.1)
# Static tick labels with accident references
source_ticks = []
source_labels = []
for name, data in accidents.items():
    if name != "Custom (user defined)":
        log_val = np.log10(data["source_Bq"])
        source_ticks.append(log_val)
        ines = data["ines"]
        label = f"INES {ines}\n{name.split(' (')[0]}"
        source_labels.append(label)
source_ticks.extend([0, 7])
source_labels.extend(["1 Bq/m³", "1e7 Bq/m³"])
source_ticks, source_labels = zip(*sorted(zip(source_ticks, source_labels)))
slider_source_log.ax.set_xticks(source_ticks)
slider_source_log.ax.set_xticklabels(source_labels, rotation=45, ha='right', fontsize=8)

# Dynamic accident label
accident_label_text = ax_source.text(1.05, 0.5, "", transform=ax_source.transAxes, fontsize=9,
                                     verticalalignment='center', bbox=dict(facecolor='white', alpha=0.7))

def update_accident_label(val):
    current_log = slider_source_log.val
    closest_name = "Custom"
    closest_diff = float('inf')
    for name, data in accidents.items():
        if name == "Custom (user defined)":
            continue
        diff = abs(np.log10(data["source_Bq"]) - current_log)
        if diff < closest_diff:
            closest_diff = diff
            closest_name = name
    if closest_diff < 0.5:
        ines = accidents[closest_name]["ines"]
        short_name = closest_name.split(' (')[0]
        label_text = f"Nearest: {short_name} (INES {ines})"
    else:
        label_text = "Custom / intermediate"
    accident_label_text.set_text(label_text)

# Height slider
ax_height = plt.axes([0.2, 0.20, 0.6, 0.03])
slider_height = Slider(ax_height, 'Height (m)', 0.0, 1000.0, valinit=initial_height, valstep=1.0)

# Rain rate slider
ax_rain = plt.axes([0.2, 0.15, 0.6, 0.03])
slider_rain = Slider(ax_rain, 'Rain rate (mm/h)', 0.0, 50.0, valinit=0.0, valstep=0.1)

# Buttons
ax_reset = plt.axes([0.7, 0.05, 0.1, 0.05])
btn_reset = Button(ax_reset, 'Reset')
ax_pause = plt.axes([0.82, 0.05, 0.1, 0.05])
btn_pause = Button(ax_pause, 'Pause')
ax_toggle = plt.axes([0.6, 0.05, 0.1, 0.05])
btn_toggle = Button(ax_toggle, 'Show Deposition')

# View state: 0 = concentration, 1 = ground deposition
view_mode = 0

paused = False
steps_per_frame = 100

def update_plot(frame):
    global paused
    if not paused:
        for _ in range(steps_per_frame):
            lib.step_simulation()
        ptr = state.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        lib.get_state(ptr)
        if view_mode == 0:
            # Concentration view
            height_m = slider_height.val
            k = int(round(height_m / DZ))
            if k < 0: k = 0
            if k >= NZ: k = NZ-1
            start = k * plane
            slice_flat = state[start:start + plane]
            slice_2d = slice_flat.reshape((NY, NX))
            slice_2d = np.maximum(slice_2d, 1e-8)
            im.set_array(slice_2d)
            cb.set_label('Concentration (Bq/m³)')
            step = lib.get_step_count()
            time_h = step * 10.0 / 3600.0
            max_val = slice_2d.max()
            rain = slider_rain.val
            if rain > 0:
                title_text.set_text(f'Time = {time_h:.1f} h, Height = {height_m:.0f} m, Rain = {rain:.1f} mm/h, Max = {max_val:.2e} Bq/m³')
            else:
                title_text.set_text(f'Time = {time_h:.1f} h, Height = {height_m:.0f} m, Max = {max_val:.2e} Bq/m³')
        else:
            # Ground deposition view
            dep_ptr = ground_dep_state.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            lib.get_ground_deposition(dep_ptr)
            dep_2d = ground_dep_state.reshape((NY, NX))
            dep_2d = np.maximum(dep_2d, 1e-8)
            im.set_array(dep_2d)
            cb.set_label('Ground Deposition (Bq/m²)')
            step = lib.get_step_count()
            time_h = step * 10.0 / 3600.0
            max_val = dep_2d.max()
            rain = slider_rain.val
            if rain > 0:
                title_text.set_text(f'Time = {time_h:.1f} h, Rain = {rain:.1f} mm/h, Deposition Max = {max_val:.2e} Bq/m²')
            else:
                title_text.set_text(f'Time = {time_h:.1f} h, Deposition Max = {max_val:.2e} Bq/m²')
    return im, title_text

def reset_sim(event):
    global paused, view_mode
    lib.finalize_simulation()
    lib.init_simulation()
    linear_strength = 10 ** slider_source_log.val
    lib.set_source_strength(linear_strength)
    lib.set_wind(slider_u.val, slider_v.val)
    lib.set_precipitation(slider_rain.val)
    view_mode = 0
    btn_toggle.label.set_text('Show Deposition')
    ptr = state.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    lib.get_state(ptr)
    height_m = slider_height.val
    k = int(round(height_m / DZ))
    if k < 0: k = 0
    if k >= NZ: k = NZ-1
    start = k * plane
    slice_flat = state[start:start + plane]
    slice_2d = slice_flat.reshape((NY, NX))
    slice_2d = np.maximum(slice_2d, 1e-8)
    im.set_array(slice_2d)
    cb.set_label('Concentration (Bq/m³)')
    step = lib.get_step_count()
    time_h = step * 10.0 / 3600.0
    max_conc = slice_2d.max()
    rain = slider_rain.val
    if rain > 0:
        title_text.set_text(f'Time = {time_h:.1f} h, Height = {height_m:.0f} m, Rain = {rain:.1f} mm/h, Max = {max_conc:.2e} Bq/m³')
    else:
        title_text.set_text(f'Time = {time_h:.1f} h, Height = {height_m:.0f} m, Max = {max_conc:.2e} Bq/m³')
    if paused:
        paused = False
        btn_pause.label.set_text('Pause')

def toggle_pause(event):
    global paused
    paused = not paused
    btn_pause.label.set_text('Resume' if paused else 'Pause')

def toggle_view(event):
    global view_mode
    view_mode = 1 - view_mode
    if view_mode == 0:
        btn_toggle.label.set_text('Show Deposition')
    else:
        btn_toggle.label.set_text('Show Concentration')

def update_wind(val):
    lib.set_wind(slider_u.val, slider_v.val)

def update_source(val):
    linear_strength = 10 ** val
    lib.set_source_strength(linear_strength)
    update_accident_label(val)

def update_height(val):
    if view_mode == 0:
        height_m = slider_height.val
        k = int(round(height_m / DZ))
        if k < 0: k = 0
        if k >= NZ: k = NZ-1
        start = k * plane
        slice_flat = state[start:start + plane]
        slice_2d = slice_flat.reshape((NY, NX))
        slice_2d = np.maximum(slice_2d, 1e-8)
        im.set_array(slice_2d)
        step = lib.get_step_count()
        time_h = step * 10.0 / 3600.0
        max_conc = slice_2d.max()
        rain = slider_rain.val
        if rain > 0:
            title_text.set_text(f'Time = {time_h:.1f} h, Height = {height_m:.0f} m, Rain = {rain:.1f} mm/h, Max = {max_conc:.2e} Bq/m³')
        else:
            title_text.set_text(f'Time = {time_h:.1f} h, Height = {height_m:.0f} m, Max = {max_conc:.2e} Bq/m³')

def update_rain(val):
    lib.set_precipitation(val)
    # Immediately update the title to reflect new rain rate
    if view_mode == 0:
        height_m = slider_height.val
        k = int(round(height_m / DZ))
        if k < 0: k = 0
        if k >= NZ: k = NZ-1
        start = k * plane
        slice_flat = state[start:start + plane]
        slice_2d = slice_flat.reshape((NY, NX))
        slice_2d = np.maximum(slice_2d, 1e-8)
        im.set_array(slice_2d)
        step = lib.get_step_count()
        time_h = step * 10.0 / 3600.0
        max_conc = slice_2d.max()
        if val > 0:
            title_text.set_text(f'Time = {time_h:.1f} h, Height = {height_m:.0f} m, Rain = {val:.1f} mm/h, Max = {max_conc:.2e} Bq/m³')
        else:
            title_text.set_text(f'Time = {time_h:.1f} h, Height = {height_m:.0f} m, Max = {max_conc:.2e} Bq/m³')
    else:
        dep_ptr = ground_dep_state.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        lib.get_ground_deposition(dep_ptr)
        dep_2d = ground_dep_state.reshape((NY, NX))
        dep_2d = np.maximum(dep_2d, 1e-8)
        im.set_array(dep_2d)
        step = lib.get_step_count()
        time_h = step * 10.0 / 3600.0
        max_val = dep_2d.max()
        if val > 0:
            title_text.set_text(f'Time = {time_h:.1f} h, Rain = {val:.1f} mm/h, Deposition Max = {max_val:.2e} Bq/m²')
        else:
            title_text.set_text(f'Time = {time_h:.1f} h, Deposition Max = {max_val:.2e} Bq/m²')

slider_u.on_changed(update_wind)
slider_v.on_changed(update_wind)
slider_source_log.on_changed(update_source)
slider_height.on_changed(update_height)
slider_rain.on_changed(update_rain)
btn_reset.on_clicked(reset_sim)
btn_pause.on_clicked(toggle_pause)
btn_toggle.on_clicked(toggle_view)

update_accident_label(slider_source_log.val)

ani = FuncAnimation(fig, update_plot, interval=33, blit=False, cache_frame_data=False)
plt.show()

lib.finalize_simulation()
