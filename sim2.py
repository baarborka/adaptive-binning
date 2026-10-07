import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ast
from pathlib import Path
import time
import shutil
from astropy.stats import bayesian_blocks

# =============================================================================
# CHANGES IN THIS VERSION
# -----------------------------------------------------------------------------
# - Adaptive (Bayesian Blocks or Equal Counts) binning is now computed 
#   INDEPENDENTLY for every energy channel.
# - Because each channel has different bin edges and a different number of bins,
#   the adaptive output is saved as a long-format CSV ('independent_adaptive_rates.csv').
# - The original fixed-width binning/plots are kept as-is for comparison.
# =============================================================================

RNG_SEED = None  # set an integer here for reproducible runs
rng = np.random.default_rng(RNG_SEED)


def format_energy_label(energy):
    if energy >= 1e6:
        return f'{energy / 1e6:.2f} GeV'
    elif energy >= 1e3:
        return f'{energy / 1e3:.2f} MeV'
    else:
        return f'{energy:.2f} keV'


flux_multiplier = 300

# ---------------------------
# Step 0: Folder creation
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

common_graphs_folder = output_base / "graphs"
common_graphs_folder.mkdir(parents=True, exist_ok=True)
run_graphs_folder = output_folder


def load_parameters(filename):
    """Parse a simple key=value params file."""
    params = {}
    with open(filename, 'r') as file:
        for line_num, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                raise ValueError(
                    f"{filename}, line {line_num}: expected 'key=value', got: {raw_line!r}"
                )
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            try:
                params[key] = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                params[key] = value
    return params


params = load_parameters('params.txt')
print(params)

alpha = params['alpha']
beta = params['beta']
E0 = params['E0']
F0 = params['F0']
E_peak = params['E_peak']
t_start = params['t_start']
t_end = params['t_end']
n_time_points = params['n_time_points']
energy_bands = params['energy_bands']
t_peak_min = params['t_peak_min']
t_peak_max = params['t_peak_max']
rise_index = params['rise_index']
decay_index = params['decay_index']

min_energy = min(E_min for E_min, _ in energy_bands)
max_energy = max(E_max for _, E_max in energy_bands)
print(min_energy, max_energy)

n_bands = len(energy_bands)
t = np.linspace(t_start, t_end, n_time_points)

avg_energies = [(E_min + E_max) / 2 for (E_min, E_max) in energy_bands]
min_avg = min(avg_energies)
max_avg = max(avg_energies)

energy_peak_times = {}
peaks = pd.DataFrame(columns=["avg_energy", "t_peak"])
for i, ((E_min, E_max), avg_energy) in enumerate(zip(energy_bands, avg_energies)):
    norm = (avg_energy - min_avg) / (max_avg - min_avg) if max_avg != min_avg else 0
    t_peak = t_peak_min + norm * (t_peak_max - t_peak_min)
    peaks.loc[i] = [avg_energy, t_peak]
    energy_peak_times[i] = t_peak


def light_curve(t_arr, t_peak, r, d):
    lc = np.zeros_like(t_arr)
    rise_phase = t_arr < t_peak
    lc[rise_phase] = (t_arr[rise_phase] / t_peak) ** r
    decay_phase = t_arr >= t_peak
    lc[decay_phase] = (t_peak / t_arr[decay_phase]) ** d * np.exp(-(t_arr[decay_phase] - t_peak) / t_peak)
    return lc


E_break = (alpha - beta) * E_peak


def band_function(E, E_peak, alpha, beta, E0, F0):
    flux = np.where(E <= E_break,
                     F0 * (E / E0) ** alpha * np.exp(-E / E_peak),
                     F0 * (E_break / E0) ** (alpha - beta) * np.exp(beta - alpha) * (E / E0) ** beta)
    return flux


def total_flux_over_spectrum(E, alpha, beta, E0, F0):
    return band_function(E, E_peak, alpha, beta, E0, F0)


def energy_to_peak_time(E):
    norm = (E - min_energy) / (max_energy - min_energy)
    return t_peak_min + norm * (t_peak_max - t_peak_min)


def shape_of_x(x, r, d):
    L = np.zeros_like(x)
    rise = x < 1
    L[rise] = x[rise] ** r
    decay = ~rise
    L[decay] = (1.0 / x[decay]) ** d * np.exp(-(x[decay] - 1.0))
    return L


eps = 1e-6
x_lo = max(eps, t_start / t_peak_max) if t_start > 0 else eps
x_hi = t_end / t_peak_min

x_grid = np.concatenate([
    np.linspace(x_lo, 1.0, 4000, endpoint=False),
    np.linspace(1.0, x_hi, 6000)
])
shape_vals = shape_of_x(x_grid, rise_index, decay_index)

dx = np.diff(x_grid)
increments = 0.5 * (shape_vals[1:] + shape_vals[:-1]) * dx
cdf_global = np.concatenate([[0.0], np.cumsum(increments)])


def sample_arrival_times(photon_t_peaks):
    x_min = t_start / photon_t_peaks if t_start > 0 else np.full_like(photon_t_peaks, eps)
    x_max = t_end / photon_t_peaks

    cdf_lo = np.interp(x_min, x_grid, cdf_global)
    cdf_hi = np.interp(x_max, x_grid, cdf_global)

    u = rng.random(len(photon_t_peaks))
    target = cdf_lo + u * (cdf_hi - cdf_lo)

    sampled_x = np.interp(target, cdf_global, x_grid)
    return sampled_x * photon_t_peaks


# ---------------------------
# Simulate individual photons
# ---------------------------
print("Generating individual photons...")

event_times = []
event_energies = []
event_bands = []

t_fine = np.linspace(t_start, t_end, 10000)

E_full = np.linspace(min_energy, max_energy, 1000000)
full_flux = total_flux_over_spectrum(E_full, alpha, beta, E0, F0)

for i, (E_min, E_max) in enumerate(energy_bands):
    mask = (E_full >= E_min) & (E_full <= E_max)
    flux_in_band = np.trapz(full_flux[mask], E_full[mask])

    t_peak_ref = energy_peak_times[i]
    lc_fine_ref = light_curve(t_fine, t_peak_ref, rise_index, decay_index)
    integral_lc = np.trapz(lc_fine_ref, t_fine)
    lambda_total = flux_in_band * integral_lc * flux_multiplier

    N_photons = rng.poisson(lambda_total)

    if N_photons > 0:
        E_sub = np.linspace(E_min, E_max, 2000)
        flux_sub = band_function(E_sub, E_peak, alpha, beta, E0, F0)
        pdf_E = flux_sub / np.trapz(flux_sub, E_sub)
        dE = np.diff(E_sub)
        cdf_E = np.concatenate([[0.0], np.cumsum(0.5 * (pdf_E[1:] + pdf_E[:-1]) * dE)])
        cdf_E /= cdf_E[-1]

        u_E = rng.random(N_photons)
        photon_energies = np.interp(u_E, cdf_E, E_sub)

        photon_t_peaks = energy_to_peak_time(photon_energies)
        arrival_times = sample_arrival_times(photon_t_peaks)

        event_times.extend(arrival_times)
        event_energies.extend(photon_energies)
        event_bands.extend([i] * N_photons)

df_events = pd.DataFrame({
    'Time (s)': event_times,
    'Energy (keV)': event_energies,
    'Energy_Band': event_bands
})
df_events = df_events.sort_values(by='Time (s)').reset_index(drop=True)

event_csv_filename = output_folder / 'photon_event_list.csv'
df_events.to_csv(event_csv_filename, index=False)
print(f"Generated {len(df_events)} total individual photons.")

# ---------------------------
# Fixed-width binning (kept for comparison)
# ---------------------------
photon_counts = np.zeros((n_bands, len(t)))

dt = (t_end - t_start) / (n_time_points - 1)
t_edges = np.linspace(t_start - dt / 2, t_end + dt / 2, n_time_points + 1)

data = {'Time (s)': t}
for i, (E_min, E_max) in enumerate(energy_bands):
    band_times = df_events[df_events['Energy_Band'] == i]['Time (s)']
    counts, _ = np.histogram(band_times, bins=t_edges)
    photon_counts[i, :] = counts / flux_multiplier
    avg_energy = (E_min + E_max) / 2
    data[avg_energy] = photon_counts[i, :]

df_photon_counts = pd.DataFrame(data)
csv_filename = output_folder / 'photon_counts_over_time.csv'
df_photon_counts.to_csv(csv_filename, index=False)

# ---------------------------
# NEW: Independent Adaptive binning per channel
# ---------------------------
ADAPTIVE_METHOD = params.get('adaptive_method', 'equal_counts')
BB_P0 = params.get('bb_p0', 0.01)
COUNTS_PER_BIN = params.get('counts_per_bin', 50)

print(f"Applying independent {ADAPTIVE_METHOD} binning to all channels...")
print(f"Config -> bb_p0: {BB_P0}, counts_per_bin: {COUNTS_PER_BIN}")

def equal_counts_edges(times, counts_per_bin):
    times_sorted = np.sort(times)
    n = len(times_sorted)
    if n < 2 * counts_per_bin:
        return np.array([times_sorted[0], times_sorted[-1]])

    chunk_starts = np.arange(counts_per_bin, n, counts_per_bin)
    if n - chunk_starts[-1] < counts_per_bin / 2:
        chunk_starts = chunk_starts[:-1]

    edges = [times_sorted[0]]
    for idx in chunk_starts:
        mid = 0.5 * (times_sorted[idx - 1] + times_sorted[idx])
        edges.append(mid)
    edges.append(times_sorted[-1])
    return np.array(edges)

if ADAPTIVE_METHOD not in ('equal_counts', 'bayesian_blocks'):
    raise ValueError("Invalid adaptive_method. Use 'equal_counts' or 'bayesian_blocks'.")

adaptive_data_list = []
independent_plot_data = [] # Stores edges/rates for the plotting step

for i, (E_min, E_max) in enumerate(energy_bands):
    band_times = df_events[df_events['Energy_Band'] == i]['Time (s)'].values
    
    if len(band_times) > 1:
        if ADAPTIVE_METHOD == 'equal_counts':
            edges = equal_counts_edges(band_times, COUNTS_PER_BIN)
        else:
            edges = bayesian_blocks(band_times, fitness='events', p0=BB_P0)
    else:
        print(f"WARNING: Band {i} has too few photons; falling back to fixed edges.")
        edges = t_edges
        
    widths = np.diff(edges)
    counts, _ = np.histogram(band_times, bins=edges)
    rates = (counts / widths) / flux_multiplier
    
    independent_plot_data.append({'edges': edges, 'rates': rates})
    
    # Store each bin as a row in long-format for the CSV
    band_df = pd.DataFrame({
        'Band_Index': i,
        'Avg_Energy': (E_min + E_max) / 2.0,
        'Bin_Start': edges[:-1],
        'Bin_End': edges[1:],
        'Rate': rates
    })
    adaptive_data_list.append(band_df)

df_adaptive = pd.concat(adaptive_data_list, ignore_index=True)
adaptive_csv_filename = output_folder / 'independent_adaptive_rates.csv'
df_adaptive.to_csv(adaptive_csv_filename, index=False)
print(f"Saved independent adaptive bins to {adaptive_csv_filename.name}")


def save_plot(category, filename_stub, fig=None):
    timestamp = int(time.time() * 1000)
    common_folder = common_graphs_folder / category
    common_folder.mkdir(parents=True, exist_ok=True)
    common_path = common_folder / f'{filename_stub}_run_{next_number}_{timestamp}.png'
    run_path = run_graphs_folder / f'{filename_stub}.png'
    (fig or plt).savefig(common_path)
    shutil.copy(common_path, run_path)
    print(f"Saved {category} plot: {common_path}")
    return common_path, run_path


# ---------------------------
# Plot 1: Combined Photon Counts (fixed-width)
# ---------------------------
n_cols = 2
n_rows = (n_bands + n_cols - 1) // n_cols

fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 6), sharex=True)
for i, (E_min, E_max) in enumerate(energy_bands):
    ax = axs[i // n_cols, i % n_cols]
    ax.plot(t, photon_counts[i, :], color='hotpink')
    label_min = format_energy_label(E_min)
    label_max = format_energy_label(E_max)
    ax.set_title(f'Photon Counts: {label_min} - {label_max}')
    ax.grid(True)
fig.supylabel('$COUNTS/cm^2/sec/keV$')
for j in range(i + 1, n_rows * n_cols):
    fig.delaxes(axs[j // n_cols, j % n_cols])
axs[-1, 0].set_xlabel('Time (s)')
axs[-1, 1].set_xlabel('Time (s)')
plt.tight_layout()
save_plot('combined_photon_counts', 'combined_photon_counts', fig=fig)

# ---------------------------
# Plot 1b: Independent Adaptive-binned rate
# ---------------------------
fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 6), sharex=True)
for i, (E_min, E_max) in enumerate(energy_bands):
    ax = axs[i // n_cols, i % n_cols]
    
    # Retrieve the specific edges and rates for this channel
    edges = independent_plot_data[i]['edges']
    rates = independent_plot_data[i]['rates']
    
    ax.step(edges[:-1], rates, where='post', color='hotpink')
    label_min = format_energy_label(E_min)
    label_max = format_energy_label(E_max)
    ax.set_title(f'{ADAPTIVE_METHOD} Rate: {label_min} - {label_max}')
    ax.grid(True)
fig.supylabel('Rate ($COUNTS/cm^2/sec/keV$)')
for j in range(i + 1, n_rows * n_cols):
    fig.delaxes(axs[j // n_cols, j % n_cols])
axs[-1, 0].set_xlabel('Time (s)')
axs[-1, 1].set_xlabel('Time (s)')
plt.tight_layout()
save_plot('independent_adaptive_rates', 'independent_adaptive_rates', fig=fig)

# ---------------------------
# Plot 2: Band Function Spectrum
# ---------------------------
E = np.logspace(0, 8, 1000)
spectrum_vals = band_function(E, E_peak, alpha, beta, E0, F0)
plt.figure(figsize=(10, 6))
plt.loglog(E, spectrum_vals, label="Band Function Spectrum", color='hotpink')
plt.xlabel("Energy (keV)")
plt.ylabel('$COUNTS/cm^2/sec/keV$')
save_plot('band_function_spectrum', 'band_function_spectrum')

# ---------------------------
# Plot 3: (Diagnostic) reference peak time vs. band-average energy
# ---------------------------
energy_band_centers = [(E_min + E_max) / 2 for E_min, E_max in energy_bands]
peak_times = [energy_peak_times[i] for i in range(len(energy_bands))]
plt.figure(figsize=(10, 6))
plt.plot(energy_band_centers, peak_times, marker='D', linestyle='-', color='hotpink')
plt.xscale('log')
plt.xlabel("Energy Band Center (keV)")
plt.ylabel("Reference Peak Time (s)")
save_plot('peak_time_vs_energy', 'peak_time_vs_energy')

# ---------------------------
# Plot 4: (Diagnostic) reference peak-time difference
# ---------------------------
lowest_peak = peaks['t_peak'].iloc[0]
peaks['Difference'] = lowest_peak - peaks['t_peak']
x = peaks['avg_energy'].values
y = peaks['Difference'].values
m, b = np.polyfit(x, y, 1)

peaks_csv_filename = output_folder / 'reference_peaks.csv'
peaks.to_csv(peaks_csv_filename, index=False)

plt.figure(figsize=(10, 6))
plt.scatter(peaks['avg_energy'], peaks['Difference'], color='hotpink', marker='D', label='Reference (band-avg) points')
x_line = np.linspace(x.min(), x.max(), 100)
y_line = m * x_line + b
plt.plot(x_line, y_line, label=f'Fit: y={m:.2e}x+{b:.2f}', color='hotpink', alpha=0.5)
plt.xscale('log')
plt.xlabel('Higher Energy Bands (keV)')
plt.ylabel('Time Shift (s)')
lowest_band = df_photon_counts.columns[1]
plt.title(f'Reference Time Shifts Relative to Lowest Energy Band: {lowest_band} keV')
plt.legend()
save_plot('peak_difference', 'peak_difference')

# ---------------------------
# Plot 5: Relative Noise Level per Energy Band (fixed-width)
# ---------------------------
plt.figure(figsize=(13, 6))
for i, (E_min, E_max) in enumerate(energy_bands):
    counts = photon_counts[i, :]
    relative_noise_percent = np.where(counts > 0, (1 / np.sqrt(counts * flux_multiplier)) * 100, np.nan)
    plt.plot(t, relative_noise_percent, label=f'{E_min}-{E_max} keV')
plt.xlabel('Time (s)')
plt.ylabel('Relative Noise Level (%)')
plt.title('Relative Noise Level Over Time for Each Energy Band')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True)
save_plot('relative_noise_percent', 'relative_noise_percent')

# ---------------------------
# Save parameters
# ---------------------------
latest_folder_file = Path('latest_folder.txt')
with open(latest_folder_file, 'w') as f:
    f.write(str(output_folder))
params_filename = output_folder / 'current_parameters.txt'
with open(params_filename, 'w') as f:
    f.write("Current Parameters:\n")
    for key, value in params.items():
        f.write(f"{key}: {value}\n")
    f.write(f"\nAdaptive binning method: {ADAPTIVE_METHOD}\n")
    if ADAPTIVE_METHOD == 'bayesian_blocks':
        f.write(f"Bayesian Blocks p0: {BB_P0}\n")
    elif ADAPTIVE_METHOD == 'equal_counts':
        f.write(f"Counts per bin: {COUNTS_PER_BIN}\n")
        
    f.write(f"\n--- Independent Bin Counts ---\n")
    for i, p_data in enumerate(independent_plot_data):
        f.write(f"Band {i}: {len(p_data['rates'])} bins\n")