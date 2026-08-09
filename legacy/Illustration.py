# Illustration.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy.optimize import curve_fit
from pathlib import Path

# Optional: Set a consistent plotting style (similar to MATLAB's default or a clean modern look)
plt.style.use('seaborn-v0_8-whitegrid') 

def fit_diffusion_model(lag_time, D, V, Z):
    """
    Diffusion model for fitting MSD data for 1D (x or y): MSD = 2 * D * t + (V*t)^2 + Z
    - D: Diffusion coefficient
    - V: Drift velocity (magnitude)
    - Z: Intercept (localization error squared, MSD_0)
    """
    return 2 * D * lag_time + (V * lag_time)**2 + Z

def fit_diffusion_model_2D(lag_time, D, V, Z):
    """
    Diffusion model for fitting MSD data for 2D (xy): MSD = 4 * D * t + (V*t)^2 + Z
    - D: Diffusion coefficient
    - V: Drift velocity (magnitude)
    - Z: Intercept (localization error squared, MSD_0)
    """
    return 4 * D * lag_time + (V * lag_time)**2 + Z

def plot_individual_msd_curves(msd_df: pd.DataFrame, output_dir: Path, video_base_name: str, pixelsize_nm: float):
    """
    Plots individual MSD curves (x, y, and xy) for each particle in separate graphs.

    Args:
        msd_df (pd.DataFrame): DataFrame containing MSD data ('particle', 'lag_time_s', 'msd_x_nm2', 'msd_y_nm2', 'msd_xy_nm2').
        output_dir (Path): Directory to save the plots.
        video_base_name (str): Base name for the output plot files (e.g., 'video_name').
        pixelsize_nm (float): The physical size of one pixel in nanometers (for converting to micrometers).
    """
    if msd_df.empty:
        print("No MSD data to plot.")
        return

    um_per_nm = 1e-3 # 1 µm = 1000 nm

    # Convert MSD from nm^2 to µm^2 for plotting
    msd_df['msd_x_um2'] = msd_df['msd_x_nm2'] * um_per_nm**2
    msd_df['msd_y_um2'] = msd_df['msd_y_nm2'] * um_per_nm**2
    msd_df['msd_xy_um2'] = msd_df['msd_xy_nm2'] * um_per_nm**2

    # Group by particle to plot each one separately
    unique_particles = msd_df['particle'].unique()

    print(f"Generating individual MSD plots for {len(unique_particles)} particles...")

    # Define common colors and labels
    colors = {'msd_x_um2': 'tab:blue', 'msd_y_um2': 'tab:red', 'msd_xy_um2': 'tab:orange'}
    labels = {'msd_x_um2': r'MSD_x', 'msd_y_um2': r'MSD_y', 'msd_xy_um2': r'MSD_{xy}'}

    for particle_id in unique_particles:
        particle_msd_df = msd_df[msd_df['particle'] == particle_id].copy() 
        
        fig, ax = plt.subplots(figsize=(8, 6))

        for msd_type in ['msd_x_um2', 'msd_y_um2', 'msd_xy_um2']:
            current_msd_series = particle_msd_df[msd_type].dropna()
            lag_times_series = particle_msd_df['lag_time_s'][current_msd_series.index]

            current_msd_values_np = current_msd_series.values
            lag_times_np = lag_times_series.values

            if len(lag_times_np) > 1:
                try:
                    fit_func = fit_diffusion_model_2D if msd_type == 'msd_xy_um2' else fit_diffusion_model
                    
                    initial_D_guess = (current_msd_series.iloc[-1] - current_msd_series.iloc[0]) / \
                                      (4 * (lag_times_series.iloc[-1] - lag_times_series.iloc[0]) + 1e-9) \
                                      if msd_type == 'msd_xy_um2' else \
                                      (current_msd_series.iloc[-1] - current_msd_series.iloc[0]) / \
                                      (2 * (lag_times_series.iloc[-1] - lag_times_series.iloc[0]) + 1e-9)
                                      
                    initial_D_guess = max(1e-12, initial_D_guess)
                    initial_Z_guess = current_msd_series.iloc[0] if current_msd_series.iloc[0] > -1e-12 else 0

                    initial_V_guess = 1e-8 
                    
                    p0 = [initial_D_guess, initial_V_guess, initial_Z_guess]
                    
                    bounds_lower = [0, 0, -0.01]
                    bounds_upper = [10.0, 10.0, 0.01] 
                    
                    popt, pcov = curve_fit(fit_func, lag_times_np, current_msd_values_np, 
                                           p0=p0, 
                                           bounds=(bounds_lower, bounds_upper),
                                           maxfev=5000)
                    
                    fitted_D, fitted_V, fitted_Z = popt
                    
                    fit_line = fit_func(lag_times_np, fitted_D, fitted_V, fitted_Z) 
                    ax.plot(lag_times_np, fit_line, color=colors[msd_type], linestyle='--', alpha=0.8, linewidth=1.5)
                    print(f"Particle {particle_id} - Fit for {msd_type}: D = {fitted_D:.4e} µm^2/s, V = {fitted_V:.4e} µm/s, Z = {fitted_Z:.4e} µm^2")

                except RuntimeError as e:
                    print(f"Particle {particle_id} - Could not fit {msd_type}: {e}")
                except ValueError as e:
                    print(f"Particle {particle_id} - ValueError during fit for {msd_type}: {e}")
                except Exception as e:
                    print(f"Particle {particle_id} - An unexpected error occurred during fit for {msd_type}: {e}")
                
            # Plot actual data points for the individual particle
            ax.plot(lag_times_np, current_msd_values_np, color=colors[msd_type], label=labels[msd_type], linewidth=2.0)

        ax.set_xlabel('Lag time (s)', fontsize=14)
        ax.set_ylabel(r'MSD ($\mu$m$^2$)', fontsize=14)
        ax.set_title(f'MSD for {video_base_name} - Particle {particle_id}', fontsize=16)
        
        ax.tick_params(axis='both', which='both', direction='in', length=6, width=1.5, color='black', labelsize=12)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('black')

        ax.legend(fontsize=12, loc='upper left', frameon=True, edgecolor='black')
        ax.grid(False)

        
        max_lag_time_for_particle = particle_msd_df['lag_time_s'].max()
        x_axis_padding = max_lag_time_for_particle * 0.05 
        ax.set_xlim(left=0, right=max_lag_time_for_particle + x_axis_padding)
        ax.set_ylim(bottom=0) 

        plt.tight_layout()
        plot_path = output_dir / f"{video_base_name}_particle_{particle_id}_msd_plot.png"
        plt.savefig(str(plot_path), dpi=300, bbox_inches='tight')
        plt.close(fig) 
        print(f"MSD plot for Particle {particle_id} saved to {plot_path}")


def plot_detected_photons_over_time(linked_particles_df: pd.DataFrame, output_dir: Path, video_base_name: str, frame_rate_hz: float, smoothing_window: int = 5):
    """
    Plots 'Detected photons/Particle/Frame' over time for each particle, showing bleaching.
    
    Args:
        linked_particles_df (pd.DataFrame): DataFrame with linked particle trajectories,
                                            must contain 'particle' and 'frame', and 'Idetect_photons_camera'.
        output_dir (Path): Directory to save the plots.
        video_base_name (str): Base name for the output plot files (e.g., 'video_name').
        frame_rate_hz (float): The frame rate of the video in Hz. Used to convert frame number to seconds.
        smoothing_window (int): Window size for the moving average smoothing.
    """
    if linked_particles_df.empty:
        print("No linked particle data to plot detected photons.")
        return
    
    if 'Idetect_photons_camera' not in linked_particles_df.columns:
        print("Error: 'Idetect_photons_camera' column not found in linked_particles_df.")
        return

    unique_particles = linked_particles_df['particle'].unique()
    print(f"Generating Detected Photons over Time plots for {len(unique_particles)} particles...")

    fig_all, ax_all = plt.subplots(figsize=(10, 7))
    
    for particle_id in unique_particles:
        # Get data for the current particle
        # MODIFIED LINE: Reset index before sorting to avoid ambiguity
        particle_data = linked_particles_df[linked_particles_df['particle'] == particle_id] \
                        .reset_index(drop=True) \
                        .sort_values(by='frame') \
                        .copy()
        
        if particle_data.empty:
            continue
        
        # Calculate elapsed time in seconds for plotting
        # Assuming frame numbers start from 1, so (frame - 1) gives 0 for the first frame.
        elapsed_time_s = (particle_data['frame'] - particle_data['frame'].min()) / frame_rate_hz 
        
        # Apply moving average smoothing
        smoothed_intensity = particle_data['Idetect_photons_camera'].rolling(window=smoothing_window, min_periods=1, center=True).mean()
        
        ax_all.plot(elapsed_time_s, smoothed_intensity, label=f'Particle {particle_id}', linewidth=1.5)

    ax_all.set_xlabel('Time (s)', fontsize=14)
    ax_all.set_ylabel('Detected photons/Particle/Frame', fontsize=14)
    ax_all.set_title(f'Detected Photons Over Time for {video_base_name}', fontsize=16)
    ax_all.tick_params(axis='both', which='major', labelsize=12)
    ax_all.legend(fontsize=10, loc='upper right', frameon=True, edgecolor='black')
    ax_all.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plot_path_all = output_dir / f"{video_base_name}_detected_photons_over_time_all_particles.png"
    plt.savefig(str(plot_path_all), dpi=300, bbox_inches='tight')
    plt.close(fig_all)
    print(f"Detected Photons Over Time plot saved to {plot_path_all}")