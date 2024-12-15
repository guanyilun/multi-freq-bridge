from dotenv import load_dotenv
load_dotenv()

import os

# General modules
from numba import jit
from schwimmbad import MPIPool
#from multiprocessing import Pool
import numpy as np
from pixell import enmap
import emcee
import itertools
import glob
import sys
import warnings
import scipy
warnings.filterwarnings('ignore')

# Pretty table
from prettytable import PrettyTable

# Custom modules
sys.path.insert(0, "src")
import utils as ut
import model

dir_base = os.getenv("DIRE_BASE")
cov_dir = os.getenv("COV_DIR")

def print_matrix_pretty(matrix):
    table = PrettyTable()
    
    # Set column names
    table.field_names = [f"Col {i+1}" for i in range(matrix.shape[1])]
    
    # Format each element in scientific notation
    for row in matrix:
        formatted_row = [f"{elem:.3e}" for elem in row]
        table.add_row(formatted_row)
    
    # Print the table
    print(table)


def main():
    global data_list_array
    global beam_list_array
    global icov
    global data_wcs
    global data_shape
    global npix
    global nmaps

    global c1_ra_min_pix, c1_ra_max_pix, c1_dec_min_pix, c1_dec_max_pix
    global c1_ra_pix, c1_dec_pix

    global c2_ra_min_pix, c2_ra_max_pix, c2_dec_min_pix, c2_dec_max_pix
    global c2_ra_pix, c2_dec_pix

    global fil_ra_min_pix, fil_ra_max_pix, fil_dec_min_pix, fil_dec_max_pix
    global fil_l0_min_pix, fil_l0_max_pix, fil_w0_min_pix, fil_w0_max_pix
    global fil_ra_pix, fil_dec_pix

    global apod_mask, mean_apod_mask

    global xgrid, ygrid
    global c1_r500_pix, c2_r500_pix

    global cf

    config_data_fname = "configs/config_mcmc.yaml"
    
    try:
        cf = ut.get_config_file(config_data_fname)
    except FileNotFoundError:
        print(f"Config file not found: {config_data_fname}")
    
    cluster_region = ut.get_region(region_center_ra=cf['region_center_ra'], 
                                   region_center_dec=cf['region_center_dec'],
                                   region_width=cf['region_width'])

    data_list, beam_list, data_wcs_list = [], [], []

    print("Appending data.")
    for _, data in enumerate(cf['data']):
        freq = data.split('_')[0]
        array = data.split('_')[1]
        inst = data.split('_')[2]
        scan = data.split('_')[3]

        if inst == 'act':
            if scan == 'dr6v2':
                data_dir = os.getenv("ACT_DATADIR")
            elif scan == 'dr6v4':
                data_dir = cf['act_data_dir_dr6v4']
            else: 
                raise ValueError("Invalid scan type")

        elif inst == 'planck':
            data_dir = os.getenv("PLANCK_DATADIR")
        
        else: 
            raise ValueError("Undefined instrument.")

        if cf['system_type'] == 'sim':
            data_str_flag = "_srcfree_model"
        elif cf['system_type'] == 'real':
            data_str_flag = "_srcfree"
        elif cf['system_type'] == 'real_with_sources':
            data_str_flag = ""
        else:
            raise ValueError("Undefined system type.")

        str_data = f"{data_dir}/*{array}*{freq}*coadd*map{data_str_flag}.fits"

        data_coadd = np.sort(glob.glob(str_data))[0]
        data_coadd = ut.imap_dim_check(enmap.read_map(data_coadd, box=cluster_region))
        
        # Convert data from uK to Jy/sr
        ff = ut.flux_factor(array, str(freq)) 
        data_coadd *= ff     
        data_wcs = data_coadd.wcs
        data_shape = data_coadd.shape

        beam_tmp = ut.get_2d_beam(freq=freq, 
                                  array=array, 
                                  inst=inst, 
                                  data_shape=data_shape, 
                                  version=str(scan),
                                  data_wcs=data_wcs)
        
        apod_mask = (enmap.apod(data_coadd*0+1, cf['apod_pix']))

        data_list.append(np.fft.fft2(data_coadd))
        beam_list.append(np.array(beam_tmp))

        data_wcs_list.append(data_wcs) 

    data_wcs = data_wcs_list[0]
    mean_apod_mask = np.mean(apod_mask)

    # Covariance appender
    print("Appending covariance.")
    combos = []
    for combo in itertools.product(cf['data'], cf['data']):
        combos.append(combo)

    covar_list = []
    covar_list_real = []
    
    for _, combo in enumerate(combos):
        # print(f"Combo: {combo}")
        freq1 = combo[0].split('_')[0]
        array1 = combo[0].split('_')[1]
        inst1 = combo[0].split('_')[2]
        scan1 = combo[0].split('_')[3]

        freq2 = combo[1].split('_')[0]
        array2 = combo[1].split('_')[1]
        inst2 = combo[1].split('_')[2]
        scan2 = combo[1].split('_')[3]

        tpsd = np.load(f"{cov_dir}/cov_{freq1}_{array1}_{inst1}_{scan1}_{freq2}_{array2}_{inst2}_{scan2}.npy")
        tpsd_real = np.fft.ifft2(tpsd).real
        
        covar_list.append(tpsd.ravel())
        covar_list_real.append(tpsd_real.ravel())

    data_list_array = np.array(data_list)
    beam_list_array = np.array(beam_list)

    npix = len(data_list_array[0].ravel())
    nmaps = len(cf['data'])

    cov = np.array(covar_list).T.reshape(npix, nmaps, nmaps)
    # cov_real = np.array(covar_list_real).T.reshape(npix, nmaps, nmaps)
    icov = np.linalg.inv(cov)
    
    for idx in range(npix):
        cov_idx = np.array( cov[idx, :, :] )
        # cov_idx_real = np.array( cov_real[idx, :, :] )

        # print_matrix_pretty(cov_idx)
        # print_matrix_pretty(cov_idx_real)

        # is_valid_real, message = ut.check_real_covariance(cov_idx_real)

        # if not is_valid_real:
        #     print(f"Real check fail: Index: {idx}")
        #     print(message)

        is_valid_complex, message = ut.check_complex_covariance(cov_idx)

        if not is_valid_complex:
            print_matrix_pretty(cov_idx)
            print(f"Complex check fail: Index: {idx}")
            print(message)