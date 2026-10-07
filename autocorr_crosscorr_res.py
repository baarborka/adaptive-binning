import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def compute_discrete_lag_correlation(lag, epsilon):
    """
    Computes the normalized discrete autocorrelation of a single residual array.
    """
    if lag == 0:
        eps_unshifted = epsilon
        eps_shifted = epsilon
    elif lag > 0:
        eps_unshifted = epsilon[:-lag]
        eps_shifted = epsilon[lag:]
    else:
        lag = abs(lag)
        eps_unshifted = epsilon[lag:]
        eps_shifted = epsilon[:-lag]
        
    covariance = np.sum(eps_unshifted * eps_shifted)
    variance = np.sum(epsilon**2)
    
    if variance == 0:
        return 0.0
        
    return covariance / variance

def compute_discrete_cross_correlation(lag, eps_ref, eps_high):
    """
    Computes the normalized discrete cross-correlation between two residual arrays.
    A positive peak indicates the high-energy noise lags (arrives after) the reference noise.
    """
    if lag == 0:
        ref_shifted = eps_ref
        high_shifted = eps_high
    elif lag > 0:
        # Shift the high-energy array backward to match an earlier reference signal
        ref_shifted = eps_ref[:-lag]
        high_shifted = eps_high[lag:]
    else:
        # Shift the high-energy array forward
        lag = abs(lag)
        ref_shifted = eps_ref[lag:]
        high_shifted = eps_high[:-lag]
        
    # Standard discrete cross-correlation of the overlapping slices
    covariance = np.sum(ref_shifted * high_shifted)
    
    # Normalize by the geometric mean of the variances of the OVERLAPPING segments
    var_ref = np.sum(ref_shifted**2)
    var_high = np.sum(high_shifted**2)
    
    if var_ref == 0 or var_high == 0:
        return 0.0
        
    return covariance / np.sqrt(var_ref * var_high)

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
    
    # Extract independent residuals for the reference band using a rolling average
    # window=5 smooths the macroscopic pulse shape out; min_periods=1 keeps the edges from becoming NaN
    smooth_ref = pd.Series(rates_ref).rolling(window=5, center=True, min_periods=1).mean().values
    eps_ref_independent = rates_ref - smooth_ref
    
    max_lag = 5
    lags = np.arange(-max_lag, max_lag + 1)
    
    all_autocorr_scores = np.zeros((n_channels, len(lags)))
    all_crosscorr_scores = np.zeros((n_channels, len(lags)))
    
    for i in range(n_channels):
        if i == reference_idx:
            continue
            
        rates_high = df[energy_bands[i]].values
        
        # --- 1. Original Autocorrelation Logic (Single combined residual) ---
        scaling_factor = np.mean(rates_high) / np.mean(rates_ref) if np.mean(rates_ref) > 0 else 0
        predicted_high = scaling_factor * rates_ref
        epsilon_combined = predicted_high - rates_high
        
        for j, lag in enumerate(lags):
            all_autocorr_scores[i, j] = compute_discrete_lag_correlation(lag, epsilon_combined)

        # --- 2. NEW Cross-Correlation Logic (Two independent residuals) ---
        smooth_high = pd.Series(rates_high).rolling(window=5, center=True, min_periods=1).mean().values
        eps_high_independent = rates_high - smooth_high
        
        for j, lag in enumerate(lags):
            all_crosscorr_scores[i, j] = compute_discrete_cross_correlation(lag, eps_ref_independent, eps_high_independent)

    # ==========================
    # Plotting
    # ==========================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    colors = plt.cm.plasma(np.linspace(0, 0.9, n_channels))
    
    for i in range(n_channels):
        if i == reference_idx:
            continue
        label = f'Band {float(energy_bands[i]):.1f} keV'
        
        # Plot 1: Autocorrelation (Original)
        ax1.plot(lags, all_autocorr_scores[i], marker='o', label=label, color=colors[i], alpha=0.85)
        
        # Plot 2: Cross-Correlation (New)
        ax2.plot(lags, all_crosscorr_scores[i], marker='s', label=label, color=colors[i], alpha=0.85)
        
    # Formatting Plot 1
    ax1.set_title('Autocorrelation of Scaled Residuals (Original)')
    ax1.set_xlabel('Lag (Number of Bins)')
    ax1.set_ylabel('Normalized Correlation $r_k$')
    ax1.set_xticks(lags)
    ax1.grid(True, alpha=0.4, linestyle='--')
    ax1.axvline(0, color='black', linewidth=1, alpha=0.3)
    
    # Formatting Plot 2
    ax2.set_title('Cross-Correlation of Independent Residuals (New)')
    ax2.set_xlabel('Lag (Number of Bins)')
    ax2.set_xticks(lags)
    ax2.grid(True, alpha=0.4, linestyle='--')
    ax2.axvline(0, color='black', linewidth=1, alpha=0.3)
    
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    
    plot_path = run_folder / 'correlation_comparison.png'
    plt.savefig(plot_path)
    plt.show()

if __name__ == "__main__":
    run_discrete_error_correlation()