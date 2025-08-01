#!/usr/bin/env python3
"""
Script to plot apodization window with proper width matching reference map
and display in RA/DEC coordinates.
"""

import numpy as np
import matplotlib.pyplot as plt
from pixell import enmap
import sys
import os
sys.path.insert(0, 'src')
import utils as ut

def plot_apodization_window(config_file, save_path=None):
    """
    Plot apodization window with width matching reference map in RA/DEC coordinates.
    
    Parameters:
    -----------
    config_file : str
        Path to configuration YAML file
    save_path : str, optional
        Path to save the plot
    """
    
    # Load configuration
    cf = ut.get_config_file(config_file)
    
    # Get region parameters from config
    region_center_ra = cf['region_center_ra']
    region_center_dec = cf['region_center_dec'] 
    region_width = cf['region_width']  # This should be 2.1 degrees
    apod_pix = cf['apod_pix']  # This should be 20 pixels
    
    # Define the region box
    region = ut.get_region(region_center_ra, region_center_dec, region_width)
    
    # Load reference map to get WCS and shape
    dire_data = "/home/gill/research/ACT/bridge/data_paper/data/data/act_no_reproj"
    ref_map_file = f"{dire_data}/act_cut_dr6v2_pa5_f098_4way_coadd_map_srcfree.fits"
    
    # Read the reference map with the region box
    ref_map = ut.imap_dim_check(enmap.read_map(ref_map_file, box=region))
    
    # Create apodization mask using the same parameters as in the analysis
    apod_mask = enmap.apod(ref_map * 0 + 1, apod_pix)
    
    # Set up the plot with proper WCS projection
    plt.rc('text', usetex=True)
    plt.rc('font', family='sans-serif', size=20)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection=ref_map.wcs)
    
    # Plot the apodization window  
    im = ax.imshow(apod_mask, origin='lower', cmap='viridis', interpolation='none',
                   vmin=0, vmax=1)
    
    # Set up coordinate axes
    ra = ax.coords[0]
    dec = ax.coords[1]
    
    ra.set_axislabel('Right Ascension')
    dec.set_axislabel('Declination')
    ra.set_major_formatter('d')
    dec.set_major_formatter('d')
    ax.invert_xaxis()
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                       fraction=0.046, pad=0.1, label='Apodization Weight')
    
    # Add title showing the parameters
    ax.set_title(f'Apodization Window\\n'
                f'Region Width: {region_width}°, Apod Pixels: {apod_pix}', 
                fontsize=16, pad=20)
    
    # Add reference map outline box for comparison
    import matplotlib.patches as patches
    
    # Calculate box corners in RA/DEC
    box_half_width = region_width / 2
    lower_left = (region_center_ra - box_half_width, region_center_dec - box_half_width)
    
    # Add rectangle showing the full region
    rect = patches.Rectangle(lower_left, region_width, region_width,
                           edgecolor='red', facecolor='none', lw=2, 
                           linestyle='--', alpha=0.8,
                           transform=ax.get_transform('world'),
                           label=f'Reference Map Region ({region_width}°×{region_width}°)')
    ax.add_patch(rect)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, format='pdf')
        print(f"Apodization window plot saved to: {save_path}")
    
    plt.show()
    
    return apod_mask, ref_map

def compare_apodization_widths(config_file, apod_pixels_list=[10, 20, 30, 40]):
    """
    Compare different apodization window widths.
    
    Parameters:
    -----------
    config_file : str
        Path to configuration YAML file  
    apod_pixels_list : list
        List of apodization pixel values to compare
    """
    
    # Load configuration
    cf = ut.get_config_file(config_file)
    region = ut.get_region(cf['region_center_ra'], cf['region_center_dec'], cf['region_width'])
    
    # Load reference map
    dire_data = "/home/gill/research/ACT/bridge/data_paper/data/data/act_no_reproj"
    ref_map_file = f"{dire_data}/act_cut_dr6v2_pa5_f098_4way_coadd_map_srcfree.fits"
    ref_map = ut.imap_dim_check(enmap.read_map(ref_map_file, box=region))
    
    # Create subplot for comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), 
                            subplot_kw={'projection': ref_map.wcs})
    axes = axes.flatten()
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='sans-serif', size=14)
    
    for i, apod_pix in enumerate(apod_pixels_list):
        ax = axes[i]
        
        # Create apodization mask
        apod_mask = enmap.apod(ref_map * 0 + 1, apod_pix)
        
        # Plot
        im = ax.imshow(apod_mask, origin='lower', cmap='viridis', 
                      interpolation='none', vmin=0, vmax=1)
        
        # Set up coordinates
        ra = ax.coords[0]
        dec = ax.coords[1]
        ra.set_axislabel('RA')
        dec.set_axislabel('Dec')
        ra.set_major_formatter('d')
        dec.set_major_formatter('d')
        ax.invert_xaxis()
        
        # Title
        ax.set_title(f'Apod Pixels: {apod_pix}', fontsize=14)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, orientation='horizontal', 
                    fraction=0.046, pad=0.1)
    
    plt.suptitle(f'Apodization Window Comparison\\n'
                f'Reference Map Width: {cf["region_width"]}°', fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Default config file
    config_file = "configs/case23_ajay.yaml"
    
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    
    print(f"Plotting apodization window for config: {config_file}")
    
    # Plot main apodization window
    apod_mask, ref_map = plot_apodization_window(
        config_file, 
        save_path="plots/apodization_window_radec.pdf"
    )
    
    # Compare different widths
    print("\\nComparing different apodization pixel values...")
    compare_apodization_widths(config_file)
    
    print("\\nApodization window analysis complete!")
    print(f"- Reference map region width: {ref_map.wcs.wcs.cdelt[0] * ref_map.shape[1] * 180/np.pi:.2f}°")
    print(f"- Apodization mask shape: {apod_mask.shape}")
    print(f"- Mean apodization weight: {np.mean(apod_mask):.3f}")