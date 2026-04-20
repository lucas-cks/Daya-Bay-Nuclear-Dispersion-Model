"""Daya Bay Nuclear Accident Plume Simulation
Requires shelter.py in the same directory.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.animation import FuncAnimation
import matplotlib.colors as colors
import os
import tkinter as tk
from tkinter import ttk

# Import the shetler.py
try:
    import shelter as plume
except ImportError:
    raise ImportError("shelter.py not found")

# import cartopy for realistic basemap 
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False
    print("Cartopy not installed")

plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

accidents = {
    "Chernobyl (1986, INES 7)": {"source_Bq": 5e6, "desc": "Major accident: large radioactive release", "ines": 7},
    "Fukushima (2011, INES 7)": {"source_Bq": 1e6, "desc": "Earthquake/tsunami led to meltdown", "ines": 7},
    "Kyshtym (1957, INES 6)": {"source_Bq": 3e5, "desc": "Waste tank explosion (Soviet Union)", "ines": 6},
    "Three Mile Island (1979, INES 5)": {"source_Bq": 5e4, "desc": "Partial core meltdown, limited release", "ines": 5},
    "Windscale Fire (1957, INES 5)": {"source_Bq": 8e4, "desc": "Reactor fire (UK)", "ines": 5},
    "Tokaimura JCO (1999, INES 4)": {"source_Bq": 5e3, "desc": "Criticality accident, worker fatalities", "ines": 4},
    "Custom (user defined)": {"source_Bq": 5.0, "desc": "Set your own source strength", "ines": None}
}

# Build the GUI dialog for selecting the accident scenario and initial wind conditions.
def show_setup_dialog():
    root = tk.Tk()
    root.title("Radioactive Plume Simulation: Select Accident Level")
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
        if selected.get() == "User defined":
            custom_frame.pack(pady=10)
        else:
            custom_frame.pack_forget()

    selected.trace_add("write", on_accident_change)
    on_accident_change()

    wind_frame = tk.Frame(root)
    tk.Label(wind_frame, text="Initial wind U (m/s, eastward):").pack(side=tk.LEFT, padx=5)
    u_entry = tk.Entry(wind_frame, width=8)
    u_entry.insert(0, "-5.0")        # westward wind (from Daya Bay to HK)
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
            u_val = -5.0
            v_val = 0.0
        result["source"] = src
        result["U"] = u_val
        result["V"] = v_val
        root.destroy()

    btn_ok = tk.Button(root, text="Start", command=on_ok, bg="green", fg="white", font=("Arial", 12))
    btn_ok.pack(pady=20)

    root.mainloop()
    return result["source"], result["U"], result["V"]

source_strength, U_init, V_init = show_setup_dialog()
print(f"Starting simulation: source strength = {source_strength:.2e} Bq/m³, U = {U_init:.1f} m/s, V = {V_init:.1f} m/s")

NX, NY, NZ = 200, 200, 30
LX, LY, LZ = 100000.0, 100000.0, 5000.0
DX, DY, DZ = LX/(NX-1), LY/(NY-1), LZ/(NZ-1)
x_km = np.linspace(0, LX/1000, NX)
y_km = np.linspace(0, LY/1000, NY)

if os.path.exists("terrain.bin"):
    terrain_z = np.fromfile("terrain.bin", dtype=np.float64).reshape(NY, NX)
    print("Loaded real terrain from terrain.bin")
    terrain_z = np.flipud(terrain_z)   # vertical flip 
else:
    CENTER_X_KM = 50.0
    CENTER_Y_KM = 50.0
    TERRAIN_HEIGHT_M = 500.0
    TERRAIN_WIDTH_M = 10000.0
    Xg, Yg = np.meshgrid(x_km, y_km)
    dx = (Xg - CENTER_X_KM) * 1000.0
    dy = (Yg - CENTER_Y_KM) * 1000.0
    r2 = dx*dx + dy*dy
    terrain_z = TERRAIN_HEIGHT_M * np.exp(-r2 / (2.0 * TERRAIN_WIDTH_M * TERRAIN_WIDTH_M))
    print("Warning: terrain.bin not found, using Gaussian mountain fallback.")

if os.path.exists("terrain_bounds.txt"):
    try:
        with open("terrain_bounds.txt", "r") as f:
            lines = f.readlines()
            lon_min = float(lines[0].split('=')[1].strip())
            lon_max = float(lines[1].split('=')[1].strip())
            lat_min = float(lines[2].split('=')[1].strip())
            lat_max = float(lines[3].split('=')[1].strip())
        print(f"Loaded geographic bounds from terrain_bounds.txt: lon {lon_min:.4f} to {lon_max:.4f}, lat {lat_min:.4f} to {lat_max:.4f}")
        use_manual = False
    except Exception as e:
        print(f"Error reading terrain_bounds.txt: {e}. Falling back to manual calibration.")
        use_manual = True
else:
    print("terrain_bounds.txt not found. Using manual calibration (anchor method).")
    use_manual = True

if use_manual:
    anchor_lon = 114.12
    anchor_lat = 22.41
    physical_width_km = 100.0
    physical_height_km = 100.0
    km_per_deg_lon = 103.0
    km_per_deg_lat = 111.0
    d_lon = (physical_width_km / 2.0) / km_per_deg_lon
    d_lat = (physical_height_km / 2.0) / km_per_deg_lat
    delta_lon = 0.0
    delta_lat = 0.0
    lon_min = anchor_lon - d_lon + delta_lon
    lon_max = anchor_lon + d_lon + delta_lon
    lat_min = anchor_lat - d_lat + delta_lat
    lat_max = anchor_lat + d_lat + delta_lat
    print(f"Using manual calibration: lon {lon_min:.3f} to {lon_max:.3f}, lat {lat_min:.3f} to {lat_max:.3f}")
else:
    anchor_lon = 114.12
    anchor_lat = 22.41

plume.init_simulation()
plume.set_source_strength(source_strength)
plume.set_wind(U_init, V_init)
plume.set_precipitation(0.0)

state = np.zeros(NX * NY * NZ, dtype=np.float64)
ground_dep_state = np.zeros(NX * NY, dtype=np.float64)
plane = NX * NY
initial_height = 100.0

if CARTOPY_AVAILABLE:
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black')
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black', alpha=0.7)
    ax.add_feature(cfeature.LAKES, edgecolor='black', facecolor='none', linewidth=0.5)
    daya_lon, daya_lat = 114.55, 22.60
    ax.plot(daya_lon, daya_lat, 'r*', markersize=10, transform=ccrs.PlateCarree(), label='Daya Bay')
    ax.plot(anchor_lon, anchor_lat, 'y*', markersize=12, transform=ccrs.PlateCarree(), label='Tai Mo Shan')
    ax.legend(loc='upper right')
    lon_grid, lat_grid = np.meshgrid(np.linspace(lon_min, lon_max, NX),
                                     np.linspace(lat_min, lat_max, NY))
    norm = colors.LogNorm(vmin=1e-8, vmax=1e7)
    img_data = np.zeros((NY, NX))
    im = ax.imshow(img_data, origin='lower', extent=[lon_min, lon_max, lat_min, lat_max],
                   cmap='jet', norm=norm, alpha=0.9, transform=ccrs.PlateCarree(),
                   interpolation='bilinear')
else:
    fig, ax = plt.subplots(figsize=(8, 8))
    norm = colors.LogNorm(vmin=1e-8, vmax=1e7)
    img_data = np.zeros((NY, NX))
    im = ax.imshow(img_data, origin='lower', extent=[0, LX/1000, 0, LY/1000],
                   cmap='jet', norm=norm, alpha=0.9, aspect='equal')
    lon_grid, lat_grid = None, None

max_height_km = np.max(terrain_z) / 1000.0
contour_levels = np.arange(0.1, max_height_km + 0.1, 0.1)

# Anchor cell info
lon_arr = np.linspace(lon_min, lon_max, NX)
lat_arr = np.linspace(lat_min, lat_max, NY)
ix = np.argmin(np.abs(lon_arr - anchor_lon))
iy = np.argmin(np.abs(lat_arr - anchor_lat))
print(f"Anchor cell: lon {lon_arr[ix]:.3f}, lat {lat_arr[iy]:.3f}, elevation {terrain_z[iy, ix]:.1f} m")

if CARTOPY_AVAILABLE:
    ax.plot(lon_min, lat_max, 'bo', markersize=8, transform=ccrs.PlateCarree(),
            label='Matrix [0,0] (top left)')
    ax.plot(lon_max, lat_min, 'go', markersize=8, transform=ccrs.PlateCarree(),
            label='Matrix [199,199] (bottom right)')
    corners_lon = [lon_min, lon_max, lon_max, lon_min]
    corners_lat = [lat_min, lat_min, lat_max, lat_max]
    ax.scatter(corners_lon, corners_lat, color='yellow', transform=ccrs.PlateCarree(),
               s=30, label='Grid corners')
    ax.legend(loc='lower left')

cb = plt.colorbar(im, ax=ax, label='Concentration (Bq/m³)')
cb.ax.tick_params(labelsize=10)
ticks = [1e-8, 1e-5, 1e-2, 1e1, 1e4, 1e7]
cb.set_ticks(ticks)
cb.set_ticklabels([f'$10^{{{int(np.log10(t))}}}$' for t in ticks])

# INES level markers
for threshold, label in [(1e6,'INES 7'), (3e5,'INES 6'), (5e4,'INES 5'), (5e3,'INES 4')]:
    pos = (np.log10(threshold) - np.log10(norm.vmin)) / (np.log10(norm.vmax) - np.log10(norm.vmin))
    if 0 < pos < 1:
        cb.ax.axhline(y=pos, color='black', linestyle='--', linewidth=1, alpha=0.7)
        cb.ax.text(0.5, pos, label, transform=cb.ax.transAxes,
                   ha='center', va='bottom', color='black', fontsize=8,
                   bbox=dict(facecolor='white', alpha=0.5, pad=1))

ax.set_xlabel('Longitude (°E)' if CARTOPY_AVAILABLE else 'x (km)')
ax.set_ylabel('Latitude (°N)' if CARTOPY_AVAILABLE else 'y (km)')
title_text = ax.set_title('Time = 0.0 h, Height = 0 m, Max = 0.00 Bq/m³')

plt.subplots_adjust(bottom=0.40)
ax_u = plt.axes([0.2, 0.35, 0.6, 0.03])
slider_u = Slider(ax_u, 'Surface U wind (m/s, W->E)', -10.0, 10.0, valinit=U_init, valstep=0.1)
ax_v = plt.axes([0.2, 0.30, 0.6, 0.03])
slider_v = Slider(ax_v, 'Surface V wind (m/s, S->N)', -10.0, 10.0, valinit=V_init, valstep=0.1)
ax_source = plt.axes([0.2, 0.25, 0.6, 0.03])
slider_source_log = Slider(ax_source, 'Source strength (log10 Bq/m³)', 0, 7, valinit=np.log10(source_strength), valstep=0.1)

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

# Update the displayed accident label to match the nearest INES source strength.
def update_accident_label(val):
    current_log = slider_source_log.val
    closest_name = "Custom"
    closest_diff = float('inf')
    for name, data in accidents.items():
        if name == "Custom (user defined)": continue
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

ax_height = plt.axes([0.2, 0.20, 0.6, 0.03])
slider_height = Slider(ax_height, 'Height (m)', 0.0, 1000.0, valinit=initial_height, valstep=1.0)
ax_rain = plt.axes([0.2, 0.15, 0.6, 0.03])
slider_rain = Slider(ax_rain, 'Rain rate (mm/h)', 0.0, 50.0, valinit=0.0, valstep=0.1)

ax_reset = plt.axes([0.7, 0.05, 0.1, 0.05])
btn_reset = Button(ax_reset, 'Reset')
ax_pause = plt.axes([0.82, 0.05, 0.1, 0.05])
btn_pause = Button(ax_pause, 'Pause')
ax_toggle = plt.axes([0.6, 0.05, 0.1, 0.05])
btn_toggle = Button(ax_toggle, 'Show Deposition')

view_mode = 0
paused = False
steps_per_frame = 100

# Advance the plume simulation and refresh the displayed concentration or deposition map.
def update_plot(frame):
    global paused
    if not paused:
        for _ in range(steps_per_frame):
            plume.step_simulation()
        plume.get_state(state)             
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
            step = plume.get_step_count()
            time_h = step * 10.0 / 3600.0
            max_val = slice_2d.max()
            rain = slider_rain.val
            if rain > 0:
                title_text.set_text(f'Time = {time_h:.1f} h, Height = {height_m:.0f} m, Rain = {rain:.1f} mm/h, Max = {max_val:.2e} Bq/m³')
            else:
                title_text.set_text(f'Time = {time_h:.1f} h, Height = {height_m:.0f} m, Max = {max_val:.2e} Bq/m³')
        else:
            plume.get_ground_deposition(ground_dep_state)
            dep_2d = ground_dep_state.reshape((NY, NX))
            dep_2d = np.maximum(dep_2d, 1e-8)
            im.set_array(dep_2d)
            cb.set_label('Ground Deposition (Bq/m²)')
            step = plume.get_step_count()
            time_h = step * 10.0 / 3600.0
            max_val = dep_2d.max()
            rain = slider_rain.val
            if rain > 0:
                title_text.set_text(f'Time = {time_h:.1f} h, Rain = {rain:.1f} mm/h, Deposition Max = {max_val:.2e} Bq/m²')
            else:
                title_text.set_text(f'Time = {time_h:.1f} h, Deposition Max = {max_val:.2e} Bq/m²')
    return im, title_text

# Reset the simulation state and update the plot with the current control settings.
def reset_sim(event):
    global paused, view_mode
    plume.finalize_simulation()
    plume.init_simulation()
    linear_strength = 10 ** slider_source_log.val
    plume.set_source_strength(linear_strength)
    plume.set_wind(slider_u.val, slider_v.val)
    plume.set_precipitation(slider_rain.val)
    view_mode = 0
    btn_toggle.label.set_text('Show Deposition')
    plume.get_state(state)
    height_m = slider_height.val
    k = int(round(height_m / DZ))
    k = max(0, min(k, NZ-1))
    start = k * plane
    slice_flat = state[start:start + plane]
    slice_2d = slice_flat.reshape((NY, NX))
    slice_2d = np.maximum(slice_2d, 1e-8)
    im.set_array(slice_2d)
    cb.set_label('Concentration (Bq/m³)')
    step = plume.get_step_count()
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

# Pause or resume the animation loop when the user clicks the button.
def toggle_pause(event):
    global paused
    paused = not paused
    btn_pause.label.set_text('Resume' if paused else 'Pause')

# Switch between concentration view and ground deposition view.
def toggle_view(event):
    global view_mode
    view_mode = 1 - view_mode
    btn_toggle.label.set_text('Show Deposition' if view_mode == 0 else 'Show Concentration')

# Apply the current wind slider values to the plume model.
def update_wind(val):
    plume.set_wind(slider_u.val, slider_v.val)

# Convert the log-scale source slider value to a linear emission rate and update the model.
def update_source(val):
    linear_strength = 10 ** val
    plume.set_source_strength(linear_strength)
    update_accident_label(val)

# Refresh the displayed vertical slice when the user changes the viewing height.
def update_height(val):
    if view_mode == 0:
        plume.get_state(state)
        height_m = slider_height.val
        k = int(round(height_m / DZ))
        k = max(0, min(k, NZ-1))
        start = k * plane
        slice_flat = state[start:start + plane]
        slice_2d = slice_flat.reshape((NY, NX))
        slice_2d = np.maximum(slice_2d, 1e-8)
        im.set_array(slice_2d)
        step = plume.get_step_count()
        time_h = step * 10.0 / 3600.0
        max_conc = slice_2d.max()
        rain = slider_rain.val
        if rain > 0:
            title_text.set_text(f'Time = {time_h:.1f} h, Height = {height_m:.0f} m, Rain = {rain:.1f} mm/h, Max = {max_conc:.2e} Bq/m³')
        else:
            title_text.set_text(f'Time = {time_h:.1f} h, Height = {height_m:.0f} m, Max = {max_conc:.2e} Bq/m³')

# Update precipitation in the plume model and refresh the current display.
def update_rain(val):
    plume.set_precipitation(val)
    if view_mode == 0:
        plume.get_state(state)
        height_m = slider_height.val
        k = int(round(height_m / DZ))
        k = max(0, min(k, NZ-1))
        start = k * plane
        slice_flat = state[start:start + plane]
        slice_2d = slice_flat.reshape((NY, NX))
        slice_2d = np.maximum(slice_2d, 1e-8)
        im.set_array(slice_2d)
        step = plume.get_step_count()
        time_h = step * 10.0 / 3600.0
        max_conc = slice_2d.max()
        if val > 0:
            title_text.set_text(f'Time = {time_h:.1f} h, Height = {height_m:.0f} m, Rain = {val:.1f} mm/h, Max = {max_conc:.2e} Bq/m³')
        else:
            title_text.set_text(f'Time = {time_h:.1f} h, Height = {height_m:.0f} m, Max = {max_conc:.2e} Bq/m³')
    else:
        plume.get_ground_deposition(ground_dep_state)
        dep_2d = ground_dep_state.reshape((NY, NX))
        dep_2d = np.maximum(dep_2d, 1e-8)
        im.set_array(dep_2d)
        step = plume.get_step_count()
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

plume.finalize_simulation()