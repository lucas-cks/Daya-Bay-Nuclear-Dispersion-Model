# plume_interactive.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.animation import FuncAnimation
import matplotlib.colors as colors
import tkinter as tk
from tkinter import ttk
import os

# Import plume_model.py contents
from plume_model import PlumeModel, NX, NY, NZ, DX, DY, DZ, plane, LX, LY, LZ, DT
from plume_model import terrain_height_gauss, TERRAIN_HEIGHT, TERRAIN_WIDTH

plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Accidents
accidents = {
    "Chernobyl (1986, INES 7)": {"source_Bq": 5e6, "desc": "Major accident – large radioactive release", "ines": 7},
    "Fukushima (2011, INES 7)": {"source_Bq": 1e6, "desc": "Earthquake/tsunami led to meltdown", "ines": 7},
    "Kyshtym (1957, INES 6)": {"source_Bq": 3e5, "desc": "Waste tank explosion (Soviet Union)", "ines": 6},
    "Three Mile Island (1979, INES 5)": {"source_Bq": 5e4, "desc": "Partial core meltdown, limited release", "ines": 5},
    "Windscale Fire (1957, INES 5)": {"source_Bq": 8e4, "desc": "Reactor fire (UK)", "ines": 5},
    "Tokaimura JCO (1999, INES 4)": {"source_Bq": 5e3, "desc": "Criticality accident, worker fatalities", "ines": 4},
    "Custom (user defined)": {"source_Bq": 5.0, "desc": "Set your own source strength", "ines": None}
}

# setup
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

# Get user choices
source_strength, U_init, V_init = show_setup_dialog()
print(f"Starting simulation: source strength = {source_strength:.2e} Bq/m³, U = {U_init:.1f} m/s, V = {V_init:.1f} m/s")

# create and initialise the model
model = PlumeModel()
model.init_simulation()
model.set_source_strength(source_strength)
model.set_wind(U_init, V_init)
model.set_precipitation(0.0)

# grid
x_km = np.linspace(0, LX/1000, NX)
y_km = np.linspace(0, LY/1000, NY)
z_m = np.linspace(0, LZ, NZ)

# Terrain visualisation
X, Y = np.meshgrid(x_km, y_km)
Z_terrain = np.zeros((NY, NX))
if model.terrain_z is not None:
    Z_terrain = model.terrain_z.reshape(NY, NX) / 1000.0   # to km
else:
    for i in range(NX):
        for j in range(NY):
            Z_terrain[j, i] = terrain_height_gauss(i*DX, j*DY) / 1000.0

# figure setup
fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(bottom=0.40)

# Initial image (zeros)
img_data = np.zeros((NY, NX))
norm = colors.LogNorm(vmin=1e-3, vmax=1e7)
im = ax.imshow(img_data, origin='lower', extent=[0, LX/1000, 0, LY/1000],
               cmap='jet', aspect='equal', norm=norm)

# Terrain contours
contour_levels = np.arange(0.1, np.max(Z_terrain)+0.1, 0.1)
ax.contour(X, Y, Z_terrain, levels=contour_levels, colors='white', linewidths=1, alpha=0.8)
ax.contourf(X, Y, Z_terrain, levels=[0, 0.5], colors='gray', alpha=0.3)

ax.set_xlabel('x (km)')
ax.set_ylabel('y (km)')
title_text = ax.set_title('Time = 0.0 h, Height = 0 m, Max = 0.00 Bq/m³')

# Colorbar
cb = plt.colorbar(im, ax=ax, label='Concentration (Bq/m³)')
cb.ax.tick_params(labelsize=10)
ticks = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
cb.set_ticks(ticks)
cb.set_ticklabels([f'$10^{{{int(np.log10(t))}}}$' for t in ticks])

# INES level markers
ines_levels = [(1e6, 'INES 7'), (3e5, 'INES 6'), (5e4, 'INES 5'), (5e3, 'INES 4')]
for threshold, label in ines_levels:
    pos = (np.log10(threshold) - np.log10(norm.vmin)) / (np.log10(norm.vmax) - np.log10(norm.vmin))
    if 0 < pos < 1:
        cb.ax.axhline(y=pos, color='white', linestyle='--', linewidth=1, alpha=0.7)
        cb.ax.text(0.5, pos, label, transform=cb.ax.transAxes,
                   ha='center', va='bottom', color='white', fontsize=8,
                   bbox=dict(facecolor='black', alpha=0.5, pad=1))

# sliders and buttons
ax_u = plt.axes([0.2, 0.35, 0.6, 0.03])
slider_u = Slider(ax_u, 'Surface U wind (m/s, W->E)', -10.0, 10.0, valinit=U_init, valstep=0.1)
ax_v = plt.axes([0.2, 0.30, 0.6, 0.03])
slider_v = Slider(ax_v, 'Surface V wind (m/s, S->N)', -10.0, 10.0, valinit=V_init, valstep=0.1)
ax_source = plt.axes([0.2, 0.25, 0.6, 0.03])
slider_source_log = Slider(ax_source, 'Source strength (log10 Bq/m³)', 0, 7, valinit=np.log10(source_strength), valstep=0.1)
ax_height = plt.axes([0.2, 0.20, 0.6, 0.03])
slider_height = Slider(ax_height, 'Height (m)', 0.0, 1000.0, valinit=100.0, valstep=1.0)
ax_rain = plt.axes([0.2, 0.15, 0.6, 0.03])
slider_rain = Slider(ax_rain, 'Rain rate (mm/h)', 0.0, 50.0, valinit=0.0, valstep=0.1)

ax_reset = plt.axes([0.7, 0.05, 0.1, 0.05])
btn_reset = Button(ax_reset, 'Reset')
ax_pause = plt.axes([0.82, 0.05, 0.1, 0.05])
btn_pause = Button(ax_pause, 'Pause')
ax_toggle = plt.axes([0.6, 0.05, 0.1, 0.05])
btn_toggle = Button(ax_toggle, 'Show Deposition')

# Source strength tick labels (accident references)
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

accident_label_text = ax_source.text(1.05, 0.5, "", transform=ax_source.transAxes, fontsize=9,
                                     verticalalignment='center', bbox=dict(facecolor='white', alpha=0.7))

# global state
view_mode = 0   # 0 = concentration, 1 = ground deposition
paused = False
steps_per_frame = 100
state = np.zeros(NX*NY*NZ, dtype=np.float64)
ground_dep_state = np.zeros(NX*NY, dtype=np.float64)

# callback functions
def update_plot(frame):
    global paused
    if not paused:
        for _ in range(steps_per_frame):
            model.step_simulation()
        model.get_state(state)
        if view_mode == 0:
            height_m = slider_height.val
            k = int(round(height_m / DZ))
            k = max(0, min(k, NZ-1))
            start = k * plane
            slice_flat = state[start:start + plane]
            slice_2d = slice_flat.reshape((NY, NX))
            slice_2d = np.maximum(slice_2d, 1e-8)
            im.set_array(slice_2d)
            cb.set_label('Concentration (Bq/m³)')
            step = model.get_step_count()
            time_h = step * DT / 3600.0
            max_val = slice_2d.max()
            rain = slider_rain.val
            if rain > 0:
                title_text.set_text(f'Time = {time_h:.1f} h, Height = {height_m:.0f} m, Rain = {rain:.1f} mm/h, Max = {max_val:.2e} Bq/m³')
            else:
                title_text.set_text(f'Time = {time_h:.1f} h, Height = {height_m:.0f} m, Max = {max_val:.2e} Bq/m³')
        else:
            model.get_ground_deposition(ground_dep_state)
            dep_2d = ground_dep_state.reshape((NY, NX))
            dep_2d = np.maximum(dep_2d, 1e-8)
            im.set_array(dep_2d)
            cb.set_label('Ground Deposition (Bq/m²)')
            step = model.get_step_count()
            time_h = step * DT / 3600.0
            max_val = dep_2d.max()
            rain = slider_rain.val
            if rain > 0:
                title_text.set_text(f'Time = {time_h:.1f} h, Rain = {rain:.1f} mm/h, Deposition Max = {max_val:.2e} Bq/m²')
            else:
                title_text.set_text(f'Time = {time_h:.1f} h, Deposition Max = {max_val:.2e} Bq/m²')
    return im, title_text

def reset_sim(event):
    global paused, view_mode
    model.finalize_simulation()
    model.init_simulation()
    linear_strength = 10 ** slider_source_log.val
    model.set_source_strength(linear_strength)
    model.set_wind(slider_u.val, slider_v.val)
    model.set_precipitation(slider_rain.val)
    view_mode = 0
    btn_toggle.label.set_text('Show Deposition')
    model.get_state(state)
    height_m = slider_height.val
    k = int(round(height_m / DZ))
    k = max(0, min(k, NZ-1))
    start = k * plane
    slice_flat = state[start:start + plane]
    slice_2d = slice_flat.reshape((NY, NX))
    slice_2d = np.maximum(slice_2d, 1e-8)
    im.set_array(slice_2d)
    cb.set_label('Concentration (Bq/m³)')
    step = model.get_step_count()
    time_h = step * DT / 3600.0
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
    btn_toggle.label.set_text('Show Deposition' if view_mode == 0 else 'Show Concentration')
    # Force immediate refresh
    update_plot(None)

def update_wind(val):
    model.set_wind(slider_u.val, slider_v.val)

def update_source(val):
    linear_strength = 10 ** val
    model.set_source_strength(linear_strength)
    # Update accident label
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

def update_height(val):
    if view_mode == 0:
        model.get_state(state)
        height_m = slider_height.val
        k = int(round(height_m / DZ))
        k = max(0, min(k, NZ-1))
        start = k * plane
        slice_flat = state[start:start + plane]
        slice_2d = slice_flat.reshape((NY, NX))
        slice_2d = np.maximum(slice_2d, 1e-8)
        im.set_array(slice_2d)
        step = model.get_step_count()
        time_h = step * DT / 3600.0
        max_conc = slice_2d.max()
        rain = slider_rain.val
        if rain > 0:
            title_text.set_text(f'Time = {time_h:.1f} h, Height = {height_m:.0f} m, Rain = {rain:.1f} mm/h, Max = {max_conc:.2e} Bq/m³')
        else:
            title_text.set_text(f'Time = {time_h:.1f} h, Height = {height_m:.0f} m, Max = {max_conc:.2e} Bq/m³')

def update_rain(val):
    model.set_precipitation(val)
    # Refresh current view
    if view_mode == 0:
        model.get_state(state)
        height_m = slider_height.val
        k = int(round(height_m / DZ))
        k = max(0, min(k, NZ-1))
        start = k * plane
        slice_flat = state[start:start + plane]
        slice_2d = slice_flat.reshape((NY, NX))
        slice_2d = np.maximum(slice_2d, 1e-8)
        im.set_array(slice_2d)
        step = model.get_step_count()
        time_h = step * DT / 3600.0
        max_conc = slice_2d.max()
        if val > 0:
            title_text.set_text(f'Time = {time_h:.1f} h, Height = {height_m:.0f} m, Rain = {val:.1f} mm/h, Max = {max_conc:.2e} Bq/m³')
        else:
            title_text.set_text(f'Time = {time_h:.1f} h, Height = {height_m:.0f} m, Max = {max_conc:.2e} Bq/m³')
    else:
        model.get_ground_deposition(ground_dep_state)
        dep_2d = ground_dep_state.reshape((NY, NX))
        dep_2d = np.maximum(dep_2d, 1e-8)
        im.set_array(dep_2d)
        step = model.get_step_count()
        time_h = step * DT / 3600.0
        max_val = dep_2d.max()
        if val > 0:
            title_text.set_text(f'Time = {time_h:.1f} h, Rain = {val:.1f} mm/h, Deposition Max = {max_val:.2e} Bq/m²')
        else:
            title_text.set_text(f'Time = {time_h:.1f} h, Deposition Max = {max_val:.2e} Bq/m²')

# Connect callbacks
slider_u.on_changed(update_wind)
slider_v.on_changed(update_wind)
slider_source_log.on_changed(update_source)
slider_height.on_changed(update_height)
slider_rain.on_changed(update_rain)
btn_reset.on_clicked(reset_sim)
btn_pause.on_clicked(toggle_pause)
btn_toggle.on_clicked(toggle_view)

# Initial accident label
update_source(slider_source_log.val)

# Run animation
ani = FuncAnimation(fig, update_plot, interval=33, blit=False, cache_frame_data=False)
plt.show()

# Cleanup
model.finalize_simulation()
