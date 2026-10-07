import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def compute_cross_correlation(tau, edges, rates_ref, rates_high):
    """
    Shifts the reference light curve by tau FIRST, maps it to the high-energy 
    channel using the adaptive sub-grid, and computes the Pearson cross-correlation.
    """
    # 1. Shift the reference timeline
    shifted_edges = edges + tau
    
    # 2. Find the exact overlapping window
    ov_start = max(shifted_edges[0], edges[0])
    ov_end = min(shifted_edges[-1], edges[-1])
    
    if ov_start >= ov_end:
        return 0.0 
        
    # 3. Create the joint sub-grid for this specific tau
    ov_edges = np.union1d(shifted_edges, edges)
    ov_edges = ov_edges[(ov_edges >= ov_start) & (ov_edges <= ov_end)]
    
    if len(ov_edges) < 2:
        return 0.0
        
    ov_widths = np.diff(ov_edges)
    ov_centers = ov_edges[:-1] + ov_widths / 2.0
    
    # 4. Map both channels to the newly aligned sub-grid
    idx_ref = np.searchsorted(shifted_edges, ov_centers) - 1
    idx_high = np.searchsorted(edges, ov_centers) - 1
    
    mapped_ref = rates_ref[idx_ref]
    mapped_high = rates_high[idx_high]
    
    # 5. Calculate the Cross-Correlation 
    numerator = np.sum(mapped_ref * mapped_high * ov_widths)
    
    # Normalize by the variance of both signals in the overlapping window
    var_ref = np.sum((mapped_ref**2) * ov_widths)
    var_high = np.sum((mapped_high**2) * ov_widths)
    
    if var_ref == 0 or var_high == 0:
        return 0.0
        
    return numerator / np.sqrt(var_ref * var_high)


def detect_delays_ccf():
    latest_file = Path('latest_folder.txt')
    if not latest_file.exists():
        print("Error: 'latest_folder.txt' not found.")
        return
        
    with open(latest_file, 'r') as f:
        run_folder = Path(f.read().strip())
        
    csv_filepath = run_folder / 'adaptive_photon_rates_over_time.csv'
    df = pd.read_csv(csv_filepath)
    
    edges = np.append(df['Bin Start (s)'].values, df['Bin End (s)'].values[-1])
    time_cols = ['Bin Start (s)', 'Bin End (s)', 'Bin Center (s)']
    energy_bands = [col for col in df.columns if col not in time_cols]
    n_channels = len(energy_bands)
    
    reference_idx = 0
    rates_ref = df[energy_bands[reference_idx]].values
    
    print(f"Loaded {n_channels} channels. Reference: {energy_bands[reference_idx]} keV.")

    # Sweep tau (adjust tau_max based on how large your simulated delay is)
    tau_min, tau_max, tau_steps = -0.1, 0.5, 1000
    taus = np.linspace(tau_min, tau_max, tau_steps)
    
    all_ccf_scores = np.zeros((n_channels, tau_steps))
    detected_delays = np.zeros(n_channels)
    
    for i in range(n_channels):
        if i == reference_idx:
            continue
            
        rates_high = df[energy_bands[i]].values
        
        for j, tau in enumerate(taus):
            all_ccf_scores[i, j] = compute_cross_correlation(tau, edges, rates_ref, rates_high)
            
        # The true delay is exactly where the cross-correlation PEAKS
        best_idx = np.argmax(all_ccf_scores[i])
        detected_delays[i] = taus[best_idx]

    # Generate Plot
    plt.figure(figsize=(12, 7))
    colors = plt.cm.plasma(np.linspace(0, 0.9, n_channels))
    
    print("\n--- Detected Delays ---")
    for i in range(n_channels):
        if i == reference_idx:
            print(f"Band {float(energy_bands[i]):.1f} keV (Ref): 0.00000 s")
            continue
            
        label = f'Band {float(energy_bands[i]):.1f} keV vs Ref'
        print(f"Band {float(energy_bands[i]):.1f} keV:       {detected_delays[i]:.5f} s")
        
        plt.plot(taus, all_ccf_scores[i], label=label, color=colors[i], alpha=0.85, linewidth=1.5)
        
        # Draw a dashed line precisely at the detected peak
        plt.axvline(detected_delays[i], color=colors[i], linestyle='--', alpha=0.6)
        
    plt.title('Cross-Correlation Function (Peaks Indicate True Delay)')
    plt.xlabel('Time Delay $\\tau$ (seconds)')
    plt.ylabel('Normalized Cross-Correlation')
    plt.grid(True, alpha=0.4)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    
    plot_path = run_folder / 'cross_correlation_delays.png'
    plt.savefig(plot_path)
    plt.show()

if __name__ == "__main__":
    detect_delays_ccf()