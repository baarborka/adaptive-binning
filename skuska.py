import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def compute_lagged_autocorrelation(tau, edges, epsilon):
    """
    Shifts the continuous residual timeline epsilon(t) by tau and computes 
    the normalized error correlation r(tau) using the sub-grid overlap.
    """
    shifted_edges = edges + tau
    
    # Define the overlapping time window
    ov_start = max(edges[0], shifted_edges[0])
    ov_end = min(edges[-1], shifted_edges[-1])
    
    if ov_start >= ov_end:
        return 0.0 
        
    # Create the sub-grid cuts for the residual overlap
    ov_edges = np.union1d(edges, shifted_edges)
    ov_edges = ov_edges[(ov_edges >= ov_start) & (ov_edges <= ov_end)]
    
    if len(ov_edges) < 2:
        return 0.0
        
    ov_widths = np.diff(ov_edges)
    ov_centers = ov_edges[:-1] + ov_widths / 2.0
    
    # Map the unshifted and shifted residuals onto the new sub-grid
    idx_unshifted = np.searchsorted(edges, ov_centers) - 1
    idx_shifted = np.searchsorted(shifted_edges, ov_centers) - 1
    
    eps_unshifted = epsilon[idx_unshifted]
    eps_shifted = epsilon[idx_shifted]
    
    # Numerator: Time-weighted average of [epsilon(t) * epsilon(t+tau)]
    covariance = np.sum(eps_unshifted * eps_shifted * ov_widths) / np.sum(ov_widths)
    
    # Denominator: Time-weighted average of [epsilon(t)^2] over the original grid
    original_widths = np.diff(edges)
    variance = np.sum((epsilon**2) * original_widths) / np.sum(original_widths)
    
    if variance == 0:
        return 0.0
        
    return covariance / variance


def run_analysis():
    # 1. Automatically locate the latest simulation folder
    latest_file = Path('latest_folder.txt')
    if not latest_file.exists():
        print("Error: 'latest_folder.txt' not found. Run the simulation script first.")
        return
        
    with open(latest_file, 'r') as f:
        run_folder = Path(f.read().strip())
        
    csv_filepath = run_folder / 'adaptive_photon_rates_over_time.csv'
    if not csv_filepath.exists():
        print(f"Error: Could not find {csv_filepath}")
        return

    # 2. Load the data
    df = pd.read_csv(csv_filepath)
    
    # Reconstruct the shared bin edges array 
    edges = np.append(df['Bin Start (s)'].values, df['Bin End (s)'].values[-1])
    
    # Extract just the energy bands (skip time columns)
    time_cols = ['Bin Start (s)', 'Bin End (s)', 'Bin Center (s)']
    energy_bands = [col for col in df.columns if col not in time_cols]
    n_channels = len(energy_bands)
    
    print(f"Loaded {n_channels} energy channels from {run_folder.name}")
    
    # 3. Setup Error-Correlation parameters
    reference_idx = 0
    tau_min, tau_max, tau_steps = -0.1, 0.1, 500
    taus = np.linspace(tau_min, tau_max, tau_steps)
    all_ec_scores = np.zeros((n_channels, tau_steps))
    
    rates_ref = df[energy_bands[reference_idx]].values
    total_time = edges[-1] - edges[0]
    mean_ref = np.sum(rates_ref * np.diff(edges)) / total_time
    
    print(f"Using Band {energy_bands[reference_idx]} keV as Reference.")

    # 4. Process all higher-energy channels
    for i in range(n_channels):
        if i == reference_idx:
            continue
            
        rates_high = df[energy_bands[i]].values
        
        # Calculate the scaling factor and residuals exactly as defined in the thesis
        mean_high = np.sum(rates_high * np.diff(edges)) / total_time
        scaling_factor_A = mean_high / mean_ref if mean_ref > 0 else 0
        
        predicted_high = scaling_factor_A * rates_ref
        epsilon = predicted_high - rates_high
        
        # Sweep tau
        for j, tau in enumerate(taus):
            all_ec_scores[i, j] = compute_lagged_autocorrelation(tau, edges, epsilon)
            
    # 5. Generate and save the Plot
    plt.figure(figsize=(12, 7))
    colors = plt.cm.viridis(np.linspace(0, 0.9, n_channels))
    
    for i in range(n_channels):
        if i == reference_idx:
            continue
        label = f'Band {float(energy_bands[i]):.1f} keV vs Ref'
        plt.plot(taus, all_ec_scores[i], label=label, color=colors[i], alpha=0.85, linewidth=1.5)
        
    plt.title('Normalized Error Correlation $r(\\tau)$ Across All Energy Channels')
    plt.xlabel('Time Delay $\\tau$ (seconds)')
    plt.ylabel('Normalized Correlation')
    plt.grid(True, alpha=0.4)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    
    # Save directly to the simulation run folder alongside the other graphs
    plot_path = run_folder / 'error_correlation_all_channels.png'
    plt.savefig(plot_path)
    print(f"Analysis complete. Plot saved to: {plot_path}")
    plt.show()

if __name__ == "__main__":
    run_analysis()