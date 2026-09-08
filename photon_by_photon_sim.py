import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ast
from pathlib import Path
import time
import shutil

def format_energy_label(energy):
    if energy >= 1e6:
        return f'{energy / 1e6:.2f} GeV'
    elif energy >= 1e3:
        return f'{energy / 1e3:.2f} MeV'
    else:
        return f'{energy:.2f} keV'

# Replaced noise_factor with flux_multiplier to scale the physical flux
flux_multiplier = 400

# ---------------------------
# Step 0: Folder creation - Create a main folder for the run.
# ---------------------------
base_name = 'n=4000_delay_noise_5p_2'
output_base = Path(base_name)
output_base.mkdir(parents=True, exist_ok=True)
existing_folders = [int(folder.name.split('_')[-1])
                    for folder in output_base.glob(f'{base_name}_*')
                    if folder.name.split('_')[-1].isdigit()]
next_number = max(existing_folders, default=0) + 1
output_folder = output_base / f'{base_name}_{next_number}'
output_folder.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Create a common "graphs" folder in the base folder (shared across runs).
# ---------------------------
common_graphs_folder = output_base / "graphs"
common_graphs_folder.mkdir(parents=True, exist_ok=True)

# ---------------------------
# For the current run, we'll also save a copy in its own folder.
# ---------------------------
run_graphs_folder = output_folder  # using the run folder itself for run-specific copies

def load_parameters(filename):
    params = {}
    with open(filename, 'r') as file:
        for line in file:
            key, value = line.strip().split('=', 1)
            try:
                params[key] = ast.literal_eval(value)
            except ValueError:
                params[key] = value
    return params

params = load_parameters('params.txt')
print(params)

alpha = params['alpha']
beta = params['beta']
E0 = params['E0']      # normally ~100 keV
F0 = params['F0']
E_peak = params['E_peak']  # normally in the formula E_0
t_start = params['t_start']
t_end = params['t_end']
n_time_points = params['n_time_points']  # number of time points
energy_bands = params['energy_bands']      # energy bands (e.g., from 1 to 10 GeV)
t_peak_min = params['t_peak_min']          # minimum peak time (s)
t_peak_max = params['t_peak_max']
rise_index = params['rise_index']
decay_index = params['decay_index']

min_energy = min(E_min for E_min, _ in energy_bands)
max_energy = max(E_max for _, E_max in energy_bands)
print(min_energy, max_energy)

E = np.logspace(0, 8, 1000)

# Step 1: Precompute average energies
avg_energies = [(E_min + E_max) / 2 for (E_min, E_max) in energy_bands]
min_avg = min(avg_energies)
max_avg = max(avg_energies)

# Step 2: Compute t_peak based on normalized avg_energy scale
energy_peak_times = {}
peaks = pd.DataFrame(columns=["avg_energy", "t_peak"])

for i, ((E_min, E_max), avg_energy) in enumerate(zip(energy_bands, avg_energies)):
    # Map avg_energy from [min_avg, max_avg] to [t_peak_min, t_peak_max]
    norm = (avg_energy - min_avg) / (max_avg - min_avg) if max_avg != min_avg else 0
    t_peak = t_peak_min + norm * (t_peak_max - t_peak_min)

    # Store
    peaks.loc[i] = [avg_energy, t_peak]
    energy_peak_times[i] = t_peak

n_bands = len(energy_bands)
t = np.linspace(t_start, t_end, n_time_points)

# ---------------------------
# Time evolution of light curve function.
# ---------------------------
def light_curve(t, t_peak, rise_index, decay_index):
    lc = np.zeros_like(t)
    # Rising phase
    rise_phase = t < t_peak
    lc[rise_phase] = (t[rise_phase] / t_peak) ** rise_index
    # Decay phase
    decay_phase = t >= t_peak
    lc[decay_phase] = (t_peak / t[decay_phase]) ** decay_index * np.exp(-(t[decay_phase] - t_peak) / t_peak)
    return lc

E_break = (alpha - beta) * E_peak
# ---------------------------
# Band function for the spectrum.
# ---------------------------
def band_function(E, E_peak, alpha, beta, E0, F0):
    flux = np.where(E <= E_break,
                    F0 * (E / E0) ** alpha * np.exp(-E / E_peak),
                    F0 * (E_break / E0) ** (alpha - beta) * np.exp(beta - alpha) * (E / E0) ** beta)
    return flux

def total_flux_over_spectrum(E, alpha, beta, E0, F0):
    return band_function(E, E_peak, alpha, beta, E0, F0)

# ---------------------------
# Simulate individual photons (Event-by-Event / Time-Tagged Events)
# ---------------------------
print("Generating individual photons...")

# Setup lists to hold our simulated individual photons
event_times = []
event_bands = []

# Create a high-resolution time grid to calculate accurate integrals and CDFs
t_fine = np.linspace(t_start, t_end, 10000)
dt_fine = t_fine[1] - t_fine[0]

# Generate the full spectrum flux once
E_full = np.linspace(min_energy, max_energy, 1000000)
full_flux = total_flux_over_spectrum(E_full, alpha, beta, E0, F0)

for i, (E_min, E_max) in enumerate(energy_bands):
    # Calculate total steady flux in this energy band
    mask = (E_full >= E_min) & (E_full <= E_max)
    flux_in_band = np.trapz(full_flux[mask], E_full[mask])
    
    # Get light curve on fine grid
    t_peak = energy_peak_times[i]
    lc_fine = light_curve(t_fine, t_peak, rise_index, decay_index)
    
    # Calculate the total expected number of photons (Lambda) 
    # over the entire observation window
    integral_lc = np.trapz(lc_fine, t_fine)
    lambda_total = flux_in_band * integral_lc * flux_multiplier
    
    # Draw the actual total number of individual photons for this band
    N_photons = np.random.poisson(lambda_total)
    
    if N_photons > 0:
        # Create the normalized Cumulative Distribution Function (CDF)
        pdf = lc_fine / np.sum(lc_fine)
        cdf = np.cumsum(pdf)
        
        # Generate N uniform random numbers between 0 and 1
        u = np.random.rand(N_photons)
        
        # Map uniform numbers through the inverse CDF to get arrival times
        arrival_times = np.interp(u, cdf, t_fine)
        
        # Store these photons
        event_times.extend(arrival_times)
        event_bands.extend([i] * N_photons)

# Create the Event List (Time-Tagged Events)
df_events = pd.DataFrame({
    'Time (s)': event_times, 
    'Energy_Band': event_bands
})
# Sort all photons chronologically
df_events = df_events.sort_values(by='Time (s)').reset_index(drop=True)

# Save the individual photon event list to CSV
event_csv_filename = output_folder / 'photon_event_list.csv'
df_events.to_csv(event_csv_filename, index=False)
print(f"Generated {len(df_events)} total individual photons.")

# ---------------------------
# Bin the photons so the rest of your plotting code works
# ---------------------------
photon_counts = np.zeros((n_bands, len(t)))

# Calculate bin edges based on the `t` array (which are bin centers)
dt = (t_end - t_start) / (n_time_points - 1)
t_edges = np.linspace(t_start - dt/2, t_end + dt/2, n_time_points + 1)

data = {'Time (s)': t}
for i, (E_min, E_max) in enumerate(energy_bands):
    # Extract times for this specific band
    band_times = df_events[df_events['Energy_Band'] == i]['Time (s)']
    
    # Bin the individual photons into the time array
    counts, _ = np.histogram(band_times, bins=t_edges)
    
    # Scale back down if matching the original unit magnitude
    photon_counts[i, :] = counts / flux_multiplier
    
    avg_energy = (E_min + E_max) / 2
    data[avg_energy] = photon_counts[i, :]

# ---------------------------
# Save binned counts to CSV.
# ---------------------------
df_photon_counts = pd.DataFrame(data)
print(df_photon_counts)
csv_filename = output_folder / 'photon_counts_over_time.csv'
df_photon_counts.to_csv(csv_filename, index=False)

# ---------------------------
# Plot 1: Combined Photon Counts.
# ---------------------------
n_cols = 2
n_rows = (n_bands + n_cols - 1) // n_cols

fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 6), sharex=True)
for i, (E_min, E_max) in enumerate(energy_bands):
    ax = axs[i // n_cols, i % n_cols]
    ax.plot(t, photon_counts[i, :], color='hotpink')#, s=5)
    label_min = format_energy_label(E_min)
    label_max = format_energy_label(E_max)
    ax.set_title(f'Photon Counts: {label_min} – {label_max}')
    #ax.set_title(f'Photon Counts for Energy Band: {E_min}-{E_max} keV')
    ax.grid(True)
fig.supylabel('$COUNTS/cm^2/sec/keV$')
for j in range(i + 1, n_rows * n_cols):
    fig.delaxes(axs[j // n_cols, j % n_cols])
axs[-1, 0].set_xlabel('Time (s)')
axs[-1, 1].set_xlabel('Time (s)')
plt.tight_layout()

# Save Plot 1 to a specific subfolder in the common graphs folder and in the run folder.
timestamp = int(time.time() * 1000)
combined_filename_common = f'combined_photon_counts_run_{next_number}_{timestamp}.png'
combined_filename_run = f'combined_photon_counts.png'
combined_common_folder = common_graphs_folder / "combined_photon_counts"
combined_common_folder.mkdir(parents=True, exist_ok=True)
combined_common_path = combined_common_folder / combined_filename_common
combined_run_path = run_graphs_folder / combined_filename_run
fig.savefig(combined_common_path)
shutil.copy(combined_common_path, combined_run_path)
print(f"Saved Combined Photon Counts plot in common folder: {combined_common_path}")
print(f"Saved Combined Photon Counts plot in run folder: {combined_run_path}")

# ---------------------------
# Plot 2: Band Function Spectrum.
# ---------------------------
counts = band_function(E, E_peak, alpha, beta, E0, F0)
plt.figure(figsize=(10, 6))
plt.loglog(E, counts, label="Band Function Spectrum", color='hotpink')
plt.xlabel("Energy (keV)")
plt.ylabel('$COUNTS/cm^2/sec/keV$')
timestamp = int(time.time() * 1000)
band_filename_common = f'band_function_spectrum_run_{next_number}_{timestamp}.png'
band_filename_run = f'band_function_spectrum.png'
band_common_folder = common_graphs_folder / "band_function_spectrum"
band_common_folder.mkdir(parents=True, exist_ok=True)
band_common_path = band_common_folder / band_filename_common
band_run_path = run_graphs_folder / band_filename_run
plt.savefig(band_common_path)
shutil.copy(band_common_path, band_run_path)
print(f"Saved Band Function Spectrum plot in common folder: {band_common_path}")
print(f"Saved Band Function Spectrum plot in run folder: {band_run_path}")

# ---------------------------
# Plot 3: Peak Time vs. Energy.
# ---------------------------
energy_band_centers = [(E_min + E_max) / 2 for E_min, E_max in energy_bands]
peak_times = [energy_peak_times[i] for i in range(len(energy_bands))]
plt.figure(figsize=(10, 6))
plt.plot(energy_band_centers, peak_times, marker='D', linestyle='-', color='hotpink')
plt.xscale('log')
plt.xlabel("Energy Band Center (keV)")
plt.ylabel("Peak Time (s)")
timestamp = int(time.time() * 1000)
peak_time_filename_common = f'peak_time_vs_energy_run_{next_number}_{timestamp}.png'
peak_time_filename_run = f'peak_time_vs_energy.png'
peak_time_common_folder = common_graphs_folder / "peak_time_vs_energy"
peak_time_common_folder.mkdir(parents=True, exist_ok=True)
peak_time_common_path = peak_time_common_folder / peak_time_filename_common
peak_time_run_path = run_graphs_folder / peak_time_filename_run
plt.savefig(peak_time_common_path)
shutil.copy(peak_time_common_path, peak_time_run_path)
print(f"Saved Peak Time vs Energy plot in common folder: {peak_time_common_path}")
print(f"Saved Peak Time vs Energy plot in run folder: {peak_time_run_path}")

# ---------------------------
# Plot 4: Peak Difference Plot.
# ---------------------------
lowest_peak = peaks['t_peak'].iloc[0]
peaks['Difference'] = lowest_peak - peaks['t_peak']
x = peaks['avg_energy'].values
y = peaks['Difference'].values

m, b = np.polyfit(x, y, 1)

print(peaks)
peaks_csv_filename = output_folder / 'posun_and_peaks_time.csv'
peaks.to_csv(peaks_csv_filename, index=False)
plt.figure(figsize=(10, 6))
plt.scatter(peaks['avg_energy'], peaks['Difference'], color='hotpink', marker='D', label='Data points')
x_line = np.linspace(x.min(), x.max(), 100)
y_line = m * x_line + b
plt.plot(x_line, y_line, label=f'Fit: y={m:.2e}x+{b:.2f}', color='hotpink', alpha=0.5)
plt.xscale('log')
plt.xlabel('Higher Energy Bands (keV)')
plt.ylabel('Time Shift (s)')
photon_counts_columns = df_photon_counts.columns[1:]
lowest_band = photon_counts_columns[0]
plt.title(f'Embeded Time Shifts Relative to Lowest Energy Band: {lowest_band} keV')
plt.legend()
timestamp = int(time.time() * 1000)
peak_diff_filename_common = f'peak_difference_run_{next_number}_{timestamp}.png'
peak_diff_filename_run = f'peak_difference.png'
peak_diff_common_folder = common_graphs_folder / "peak_difference"
peak_diff_common_folder.mkdir(parents=True, exist_ok=True)
peak_diff_common_path = peak_diff_common_folder / peak_diff_filename_common
peak_diff_run_path = run_graphs_folder / peak_diff_filename_run
plt.savefig(peak_diff_common_path)
shutil.copy(peak_diff_common_path, peak_diff_run_path)
print(f"Saved Peak Difference plot in common folder: {peak_diff_common_path}")
print(f"Saved Peak Difference plot in run folder: {peak_diff_run_path}")

# ---------------------------
# Plot 5: Relative Noise Level in % for Each Energy Band
# ---------------------------
plt.figure(figsize=(13, 6))

for i, (E_min, E_max) in enumerate(energy_bands):
    counts = photon_counts[i, :]

    # Avoid division by zero
    relative_noise_percent = np.where(counts > 0, (1 / np.sqrt(counts * flux_multiplier)) * 100, np.nan)

    plt.plot(t, relative_noise_percent, label=f'{E_min}-{E_max} keV')

plt.xlabel('Time (s)')
plt.ylabel('Relative Noise Level (%)')
plt.title('Relative Noise Level Over Time for Each Energy Band')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True)

# Save Plot 5
timestamp = int(time.time() * 1000)
relative_noise_filename_common = f'relative_noise_percent_run_{next_number}_{timestamp}.png'
relative_noise_filename_run = f'relative_noise_percent.png'

relative_noise_common_folder = common_graphs_folder / "relative_noise_percent"
relative_noise_common_folder.mkdir(parents=True, exist_ok=True)
relative_noise_common_path = relative_noise_common_folder / relative_noise_filename_common
relative_noise_run_path = run_graphs_folder / relative_noise_filename_run
plt.savefig(relative_noise_common_path)
shutil.copy(relative_noise_common_path, relative_noise_run_path)

print(f"Saved Relative Noise Percent plot in common folder: {relative_noise_common_path}")
print(f"Saved Relative Noise Percent plot in run folder: {relative_noise_run_path}")

# ---------------------------
# Save current parameters and update latest_folder.txt.
# ---------------------------
latest_folder_file = Path('latest_folder.txt')
with open(latest_folder_file, 'w') as f:
    f.write(str(output_folder))
params_filename = output_folder / 'current_parameters.txt'
with open(params_filename, 'w') as f:
    f.write("Current Parameters:\n")
    for key, value in params.items():
        f.write(f"{key}: {value}\n")