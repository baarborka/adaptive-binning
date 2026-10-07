import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def compute_discrete_lag_correlation(lag, epsilon):
    """
    Computes the normalized discrete autocorrelation of the residual array.
    Shifts strictly by array index, ignoring physical bin widths.
    """
    if lag == 0:
        eps_unshifted = epsilon
        eps_shifted = epsilon
    elif lag > 0:
        # Shift the array forward by 'lag' indices
        eps_unshifted = epsilon[:-lag]
        eps_shifted = epsilon[lag:]
    else:
        # Shift the array backward by 'lag' indices
        lag = abs(lag)
        eps_unshifted = epsilon[lag:]
        eps_shifted = epsilon[:-lag]
        
    # Standard discrete correlation of the overlapping array slices
    covariance = np.sum(eps_unshifted * eps_shifted)
    variance = np.sum(epsilon**2)
    
    if variance == 0:
        return 0.0
        
    return covariance / variance

def run_discrete_error_correlation():
    latest_file = Path('latest_folder.txt')
    if not latest_file.exists():
        print("Error: 'latest_folder.txt' not found.")
        return
        
    with open(latest_file, 'r') as f:
        run_folder = Path(f.read().strip())
        
    csv_filepath = run_folder / 'adaptive_photon_rates_over_time.csv'
    df = pd.read_csv(csv_filepath)
    
    time_cols = ['Bin Start (s)', 'Bin End (s)', 'Bin Center (s)']
    energy_bands = [col for col in df.columns if col not in time_cols]
    n_channels = len(energy_bands)
    
    reference_idx = 0
    rates_ref = df[energy_bands[reference_idx]].values
    
    # We will look at shifts of up to 5 bins in either direction
    max_lag = 5
    lags = np.arange(-max_lag, max_lag + 1)
    
    all_discrete_scores = np.zeros((n_channels, len(lags)))
    
    for i in range(n_channels):
        if i == reference_idx:
            continue
            
        rates_high = df[energy_bands[i]].values
        
        # 1. Scale the reference to match the high-energy flux (simple array mean)
        scaling_factor = np.mean(rates_high) / np.mean(rates_ref) if np.mean(rates_ref) > 0 else 0
        predicted_high = scaling_factor * rates_ref
        
        # 2. Calculate the static prediction error timeline e(t)
        epsilon = predicted_high - rates_high
        
        # 3. Shift by discrete bin index and correlate
        for j, lag in enumerate(lags):
            all_discrete_scores[i, j] = compute_discrete_lag_correlation(lag, epsilon)

    # Generate Plot
    plt.figure(figsize=(10, 6))
    colors = plt.cm.plasma(np.linspace(0, 0.9, n_channels))
    
    for i in range(n_channels):
        if i == reference_idx:
            continue
        label = f'Band {float(energy_bands[i]):.1f} keV vs Ref'
        
        # Plotting discrete points connected by lines
        plt.plot(lags, all_discrete_scores[i], marker='o', label=label, color=colors[i], alpha=0.85, linewidth=1.5)
        
    plt.title('Discrete Error Correlation (Bin Index Lags)')
    plt.xlabel('Lag (Number of Bins)')
    plt.ylabel('Normalized Correlation $r_k$')
    
    # Force the x-axis to show integer bin steps
    plt.xticks(lags)
    plt.grid(True, alpha=0.4, linestyle='--')
    plt.axvline(0, color='black', linewidth=1, alpha=0.3)
    
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    
    plot_path = run_folder / 'discrete_error_correlation.png'
    plt.savefig(plot_path)
    plt.show()

if __name__ == "__main__":
    run_discrete_error_correlation()