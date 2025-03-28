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
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

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

def get_initial_params(cf, n_walkers):
    
    # cluster 1: Abell 401
    c1_ra_init = np.random.uniform(c1_ra_min_pix, c1_ra_max_pix, size=n_walkers)
    c1_dec_init = np.random.uniform(c1_dec_min_pix, c1_dec_max_pix, size=n_walkers)
    c1_beta_init = np.random.uniform(cf['c1_beta_min'], cf['c1_beta_max'], size=n_walkers)
    c1_rc_arcmin_init = np.random.uniform(cf['c1_rc_arcmin_min'], cf['c1_rc_arcmin_max'], size=n_walkers)
    c1_R_init = np.random.uniform(cf['c1_R_min'], cf['c1_R_max'], size=n_walkers)
    c1_Dtau_init = np.random.uniform(cf['c1_Dtau_min'], cf['c1_Dtau_max'], size=n_walkers)
    c1_Te_init = np.random.uniform(cf['c1_Te_min'], cf['c1_Te_max'], size=n_walkers)
    c1_theta_init = np.random.uniform(cf['c1_theta_min'], cf['c1_theta_max'], size=n_walkers)
    c1_A_D_init = np.random.uniform(float(cf['c1_A_D_min']), float(cf['c1_A_D_max']), size=n_walkers)
   # c1_v_init = np.random.uniform(cf['c1_v_min'], cf['c1_v_max'], size=n_walkers)

    # cluster 2: Abell 399
    c2_ra_init = np.random.uniform(c2_ra_min_pix, c2_ra_max_pix, size=n_walkers)
    c2_dec_init = np.random.uniform(c2_dec_min_pix, c2_dec_max_pix, size=n_walkers)
    c2_beta_init = np.random.uniform(cf['c2_beta_min'], cf['c2_beta_max'], size=n_walkers)
    c2_rc_arcmin_init = np.random.uniform(cf['c2_rc_arcmin_min'], cf['c2_rc_arcmin_max'], size=n_walkers)
    c2_R_init = np.random.uniform(cf['c2_R_min'], cf['c2_R_max'], size=n_walkers)
    c2_Dtau_init = np.random.uniform(cf['c2_Dtau_min'], cf['c2_Dtau_max'], size=n_walkers)
    c2_Te_init = np.random.uniform(cf['c2_Te_min'], cf['c2_Te_max'], size=n_walkers)
    c2_theta_init = np.random.uniform(cf['c2_theta_min'], cf['c2_theta_max'], size=n_walkers)
    c2_A_D_init = np.random.uniform(float(cf['c2_A_D_min']), float(cf['c2_A_D_max']), size=n_walkers)
    #c2_v_init = np.random.uniform(cf['c2_v_min'], cf['c2_v_max'], size=n_walkers)

    # filament
    fil_ra_init = np.random.uniform(fil_ra_min_pix, fil_ra_max_pix, size=n_walkers)
    fil_dec_init = np.random.uniform(fil_dec_min_pix, fil_dec_max_pix, size=n_walkers)
    fil_l0_init = np.random.uniform(fil_l0_min_pix,fil_l0_max_pix, size=n_walkers)
    fil_w0_init = np.random.uniform(fil_w0_min_pix, fil_w0_max_pix, size=n_walkers)
    fil_Dtau_init = np.random.uniform(cf['fil_Dtau_min'], cf['fil_Dtau_max'], size=n_walkers)
    fil_Te_init = np.random.uniform(cf['fil_Te_min'], cf['fil_Te_max'], size=n_walkers)
    fil_A_D_init = np.random.uniform(float(cf['fil_A_D_min']), float(cf['fil_A_D_max']), size=n_walkers)
    #fil_v_init = np.random.uniform(cf['fil_v_min'], cf['fil_v_max'], size=n_walkers)

    # average velocity
    vavg_init = np.random.uniform(cf['v_avg_min'], cf['v_avg_max'], size=n_walkers)

    if cf["model_choice"] == "fit_vavg":
        # fits average velocity only
        ret_array = np.array([
            # Cluster 1: Abell 401
            c1_ra_init,         # 0 - Right Ascension
            c1_dec_init,        # 1 - Declination
            c1_beta_init,       # 2 - Beta
            c1_rc_arcmin_init,  # 3 - Core radius (arcminutes)
            c1_R_init,          # 4 - Radius
            c1_theta_init,      # 5 - Theta
            c1_Dtau_init,       # 6 - Optical depth
            c1_Te_init,         # 7 - Electron temperature
            c1_A_D_init,        # 8 - Amplitude of dust

            # Cluster 2: Abell 399
            c2_ra_init,         # 9 - Right Ascension
            c2_dec_init,        # 10 - Declination
            c2_beta_init,       # 11 - Beta
            c2_rc_arcmin_init,  # 12 - Core radius (arcminutes)
            c2_R_init,          # 13 - Radius
            c2_theta_init,      # 14 - Theta
            c2_Dtau_init,       # 15 - Optical depth
            c2_Te_init,         # 16 - Electron temperature
            c2_A_D_init,        # 17 - Amplitude of dust

            # Filament
            fil_ra_init,        # 18 - Right Ascension
            fil_dec_init,       # 19 - Declination
            fil_l0_init,        # 20 - Length (pixels)
            fil_w0_init,        # 21 - Width (pixels)
            fil_Dtau_init,      # 22 - Optical depth
            fil_Te_init,        # 23 - Electron temperature
            fil_A_D_init,       # 24 - Amplitude of dust

            vavg_init           # 25 - Velocity

        ]).T

        return ret_array

    elif cf["model_choice"] == "fit_individual":    
        # fits individual cluster and filament velocities    
        # Define the initial parameter array with indices
        ret_array = np.array([

        # Cluster 1: Abell 401
        c1_ra_init,         # 0 - Right Ascension
        c1_dec_init,        # 1 - Declination
        c1_beta_init,       # 2 - Beta
        c1_rc_arcmin_init,  # 3 - Core radius (arcminutes)
        c1_R_init,          # 4 - Radius
        c1_theta_init,      # 5 - Theta
        c1_Dtau_init,       # 6 - Optical depth
        c1_Te_init,         # 7 - Electron temperature
        c1_A_D_init,        # 8 - Amplitude of dust
        c1_v_init,          # 9 - Velocity

        # Cluster 2: Abell 399
        c2_ra_init,         # 10 - Right Ascension
        c2_dec_init,        # 11 - Declination
        c2_beta_init,       # 12 - Beta
        c2_rc_arcmin_init,  # 13 - Core radius (arcminutes)
        c2_R_init,          # 14 - Radius
        c2_theta_init,      # 15 - Theta
        c2_Dtau_init,       # 16 - Optical depth
        c2_Te_init,         # 17 - Electron temperature
        c2_A_D_init,        # 18 - Amplitude of dust
        c2_v_init,          # 19 - Velocity

        # Filament
        fil_ra_init,        # 20 - Right Ascension
        fil_dec_init,       # 21 - Declination
        fil_l0_init,        # 22 - Length (pixels)
        fil_w0_init,        # 23 - Width (pixels)
        fil_Dtau_init,      # 24 - Optical depth
        fil_Te_init,        # 25 - Electron temperature
        fil_A_D_init,       # 26 - Amplitude of dust

        fil_v_init          # 27 - Velocity

        ]).T
        return ret_array
    
    else:
        raise ValueError("Undefined model choice. Choose either 'fit_vavg' or 'fit_individual' in the config file.")

def lnprior(theta):  

    if cf["model_choice"] == "fit_vavg":  
        check_c1 = (c1_ra_min_pix < theta[0] < c1_ra_max_pix
                    and c1_dec_min_pix < theta[1] < c1_dec_max_pix
                    and cf['c1_beta_min'] < theta[2] < cf['c1_beta_max']
                    and cf['c1_rc_arcmin_min'] < theta[3] < cf['c1_rc_arcmin_max']
                    and cf['c1_R_min'] < theta[4] < cf['c1_R_max']
                    and cf['c1_theta_min'] < theta[5] < cf['c1_theta_max']
                    and cf['c1_Dtau_min'] < theta[6] < cf['c1_Dtau_max']
                    and cf['c1_Te_min'] < theta[7] < cf['c1_Te_max']
                    and cf['c1_A_D_min'] < theta[8] < cf['c1_A_D_max'])
        
        check_c2 = (c2_ra_min_pix < theta[9] < c2_ra_max_pix
                    and c2_dec_min_pix < theta[10] < c2_dec_max_pix
                    and cf['c2_beta_min'] < theta[11] < cf['c2_beta_max']
                    and cf['c2_rc_arcmin_min'] < theta[12] < cf['c2_rc_arcmin_max']
                    and cf['c2_R_min'] < theta[13] < cf['c2_R_max']
                    and cf['c2_theta_min'] < theta[14] < cf['c2_theta_max']
                    and cf['c2_Dtau_min'] < theta[15] < cf['c2_Dtau_max']
                    and cf['c2_Te_min'] < theta[16] < cf['c2_Te_max']
                    and cf['c2_A_D_min'] < theta[17] < cf['c2_A_D_max'])
        
        check_fil = (fil_ra_min_pix < theta[18] < fil_ra_max_pix
                    and fil_dec_min_pix < theta[19] < fil_dec_max_pix
                    and fil_l0_min_pix < theta[20] < fil_l0_max_pix
                    and fil_w0_min_pix < theta[21] < fil_w0_max_pix
                    and cf['fil_Dtau_min'] < theta[22] < cf['fil_Dtau_max']
                    and cf['fil_Te_min'] < theta[23] < cf['fil_Te_max']
                    and cf['fil_A_D_min'] < theta[24] < cf['fil_A_D_max']
                    )
        
        check_v_avg = cf['v_avg_min'] < theta[25] < cf['v_avg_max']
        
        if check_c1 and check_c2 and check_v_avg and check_fil:
            term1 = -0.5 * ( (theta[7]-cf['c1_Te_mean'])**2. / cf['c1_Te_std']**2 )
            term2 = -0.5 * ( (theta[16]-cf['c2_Te_mean'])**2. / cf['c2_Te_std']**2 )
            term3 = -0.5 * ( (theta[23]-cf['fil_Te_mean'])**2. / cf['fil_Te_std']**2 )
            return term1 + term2 + term3
        else:
            return -np.inf
    
    elif cf["model_choice"] == "fit_individual":
        check_c1 = (c1_ra_min_pix < theta[0] < c1_ra_max_pix
                    and c1_dec_min_pix < theta[1] < c1_dec_max_pix
                    and cf['c1_beta_min'] < theta[2] < cf['c1_beta_max']
                    and cf['c1_rc_arcmin_min'] < theta[3] < cf['c1_rc_arcmin_max']
                    and cf['c1_R_min'] < theta[4] < cf['c1_R_max']
                    and cf['c1_theta_min'] < theta[5] < cf['c1_theta_max']
                    and cf['c1_Dtau_min'] < theta[6] < cf['c1_Dtau_max']
                    and cf['c1_Te_min'] < theta[7] < cf['c1_Te_max']
                    and cf['c1_A_D_min'] < theta[8] < cf['c1_A_D_max']
                    and cf['c1_v_min'] < theta[9] < cf['c1_v_max'])
        
        check_c2 = (c2_ra_min_pix < theta[10] < c2_ra_max_pix
                    and c2_dec_min_pix < theta[11] < c2_dec_max_pix
                    and cf['c2_beta_min'] < theta[12] < cf['c2_beta_max']
                    and cf['c2_rc_arcmin_min'] < theta[13] < cf['c2_rc_arcmin_max']
                    and cf['c2_R_min'] < theta[14] < cf['c2_R_max']
                    and cf['c2_theta_min'] < theta[15] < cf['c2_theta_max']
                    and cf['c2_Dtau_min'] < theta[16] < cf['c2_Dtau_max']
                    and cf['c2_Te_min'] < theta[17] < cf['c2_Te_max']
                    and cf['c2_A_D_min'] < theta[18] < cf['c2_A_D_max']
                    and cf['c2_v_min'] < theta[19] < cf['c2_v_max'])
        
        check_fil = (fil_ra_min_pix < theta[20] < fil_ra_max_pix
                    and fil_dec_min_pix < theta[21] < fil_dec_max_pix
                    and fil_l0_min_pix < theta[22] < fil_l0_max_pix
                    and fil_w0_min_pix < theta[23] < fil_w0_max_pix
                    and cf['fil_Dtau_min'] < theta[24] < cf['fil_Dtau_max']
                    and cf['fil_Te_min'] < theta[25] < cf['fil_Te_max']
                    and cf['fil_A_D_min'] < theta[26] < cf['fil_A_D_max']
                    and cf['fil_v_min'] < theta[27] < cf['fil_v_max'])
        
        if check_c1 and check_c2 and check_fil:
            term1 = -0.5 * ( (theta[7]-cf['c1_Te_mean'])**2. / cf['c1_Te_std']**2 )
            term2 = -0.5 * ( (theta[17]-cf['c2_Te_mean'])**2. / cf['c2_Te_std']**2 )
            term3 = -0.5 * ( (theta[25]-cf['fil_Te_mean'])**2. / cf['fil_Te_std']**2 )
            return term1 + term2 + term3
        else:
            return -np.inf

@jit(nopython=True, parallel=False)
def lnlike_loop(resid, icov):
    
    like_loop = 0
    
    for idx in range(npix):
        # norm = np.log(np.linalg.det(icov[idx, :, :])).real
        like_loop += (-0.5 * (np.conj(resid[idx, :]) @ icov[idx, :, :] @ resid[idx, :].T)).real
        # like_loop += -0.5 * norm
    
    return like_loop

def lnlike(theta):
    """
    Likelihood function for MCMC.

    Parameters
    ----------
    theta : array

    Returns
    -------
    float : likelihood
    """
    
    c1 = model.Cluster(theta=theta, name="abell401", model_choice=cf["model_choice"])
    c2 = model.Cluster(theta=theta, name="abell399", model_choice=cf["model_choice"])
    fil = model.Filament(theta=theta, model_choice=cf["model_choice"])
    
    resids = []

    for idx in range(len(cf['data'])):
        data_str = cf['data'][idx]
        freq = float(data_str.split('_')[0])
        array = data_str.split('_')[1]
        #inst = data_str.split('_')[2]
        #scan = data_str.split('_')[3]
        
        c1_model = c1.szmodel(frequency=freq, 
                              array=array, 
                              z=cf['c1_z'],
                              muo=cf['c1_muo'],
                              xgrid=xgrid, 
                              ygrid=ygrid,
                              ellipticity_type=cf["ellipticity_type"])
        
        c2_model = c2.szmodel(frequency=freq,
                                array=array,
                                z=cf['c2_z'],
                                muo=cf['c2_muo'],
                                xgrid=xgrid,
                                ygrid=ygrid, 
                                ellipticity_type=cf["ellipticity_type"])
        
        fil_model = fil.szmodel(frequency=freq,
                                array=array,
                                xgrid=xgrid,
                                ygrid=ygrid, 
                                z=cf['fil_z'],
                                muo=cf['fil_muo'])
        
        fit_model = np.fft.fft2(c1_model + c2_model + fil_model) * beam_list_array[idx]
        
        resid_loop = data_list_array[idx].ravel() - fit_model.ravel()
        
        resids.append(resid_loop)

    resid = np.array(resids).T
    
    return lnlike_loop(resid, icov)

def lnprob(theta):
    lp = lnprior(theta=theta)
    
    if not np.isfinite(lp):
        return -np.inf
    
    return lp + lnlike(theta=theta)    

def main(config_data_fname, cov_dir=cov_dir, outdir=dir_base):
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
    
    # Testing
    test_cov = False

    if test_cov:
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
                sys.exit(-1)

    print("All good!")

    # Testing

    # temp
    # covar_array = np.array(covar_list).T.reshape(npix, nmaps, nmaps)
    # icov = np.zeros_like(covar_array)

    # for i in range(npix):
    #     U, s, Vh = scipy.linalg.svd(covar_array[i])
    #     s_inv = 1 / s
    #     # icov[i] = np.dot(Vh.T @ np.diag(s_inv), U.T)
    #     icov[i] = Vh.T @ np.diag(s_inv) @ U.T
    # # end temp

    xgrid, ygrid = np.meshgrid(np.arange(0, data_shape[1], 1), 
                               np.arange(0, data_shape[0], 1))
    
    # Deg to pix
    c1_r500_pix = ut.get_r500(cf['c1_z'], cf['c1_mass'])
    c1_ra_min_pix = data_wcs.celestial.wcs_world2pix(cf['c1_ra_min'], cf['c1_dec_min'], 0)[0]
    c1_ra_max_pix = data_wcs.celestial.wcs_world2pix(cf['c1_ra_max'], cf['c1_dec_max'], 0)[0]
    c1_dec_min_pix = data_wcs.celestial.wcs_world2pix(cf['c1_ra_min'], cf['c1_dec_min'], 0)[1]
    c1_dec_max_pix = data_wcs.celestial.wcs_world2pix(cf['c1_ra_max'], cf['c1_dec_max'], 0)[1]
    c1_ra_pix = data_wcs.celestial.wcs_world2pix(cf['c1_ra'], cf['c1_dec'], 0)[0]
    c1_dec_pix = data_wcs.celestial.wcs_world2pix(cf['c1_ra'], cf['c1_dec'], 0)[1]

    c2_r500_pix = ut.get_r500(cf['c2_z'], cf['c2_mass'])
    c2_ra_min_pix = data_wcs.celestial.wcs_world2pix(cf['c2_ra_min'], cf['c2_dec_min'], 0)[0]
    c2_ra_max_pix = data_wcs.celestial.wcs_world2pix(cf['c2_ra_max'], cf['c2_dec_max'], 0)[0]
    c2_dec_min_pix = data_wcs.celestial.wcs_world2pix(cf['c2_ra_min'], cf['c2_dec_min'], 0)[1]
    c2_dec_max_pix = data_wcs.celestial.wcs_world2pix(cf['c2_ra_max'], cf['c2_dec_max'], 0)[1]
    c2_ra_pix = data_wcs.celestial.wcs_world2pix(cf['c2_ra'], cf['c2_dec'], 0)[0]
    c2_dec_pix = data_wcs.celestial.wcs_world2pix(cf['c2_ra'], cf['c2_dec'], 0)[1]

    fil_ra_min_pix = data_wcs.celestial.wcs_world2pix(cf['fil_ra_min'], cf['fil_dec_min'], 0)[0]
    fil_ra_max_pix = data_wcs.celestial.wcs_world2pix(cf['fil_ra_max'], cf['fil_dec_max'], 0)[0]
    fil_dec_min_pix = data_wcs.celestial.wcs_world2pix(cf['fil_ra_min'], cf['fil_dec_min'], 0)[1]
    fil_dec_max_pix = data_wcs.celestial.wcs_world2pix(cf['fil_ra_max'], cf['fil_dec_max'], 0)[1]
    fil_ra_pix = data_wcs.celestial.wcs_world2pix(cf['fil_ra'], cf['fil_dec'], 0)[0]
    fil_dec_pix = data_wcs.celestial.wcs_world2pix(cf['fil_ra'], cf['fil_dec'], 0)[1]

    fil_l0_min_pix = cf['fil_l0_min_arcmin'] / 0.5  # arcmin/pix
    fil_l0_max_pix = cf['fil_l0_max_arcmin'] / 0.5  # arcmin/pix
    fil_w0_min_pix = cf['fil_w0_min_arcmin'] / 0.5  # arcmin/pix
    fil_w0_max_pix = cf['fil_w0_max_arcmin'] / 0.5  # arcmin/pix 

    mcmc_filepath = f"{outdir}/{cf['mcmc_filename']}"
    if rank == 0:
        if not os.path.exists(f"{outdir}"):
            os.makedirs(f"{outdir}")
    comm.Barrier()

    # save config_data to run_name directory
    if rank == 0:
        config_file_path = f"{outdir}/config.yaml"
        ut.save_config_file(config_file_path, cf)

    print("MCMC file path: ", mcmc_filepath)

    if os.path.exists(mcmc_filepath) and cf['repeat_h5'] == 'true':
        print("MCMC file exists.")
        init_params = None
    else:
        init_params = get_initial_params(cf=cf, n_walkers=cf['n_walkers'])
    
    n_dim = get_initial_params(cf=cf, n_walkers=cf['n_walkers']).shape[1]
    backend = emcee.backends.HDFBackend(mcmc_filepath)
    
    print("Running MCMC.")

    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        
        sampler = emcee.EnsembleSampler(nwalkers=cf['n_walkers'], 
                                        ndim=n_dim, 
                                        log_prob_fn=lnprob, 
                                        pool=pool, 
                                        backend=backend)

        # sampler.run_mcmc(initial_state=init_params, nsteps=cf['n_iter'])
        nsteps = cf['n_iter']
        for i, result in enumerate(sampler.sample(init_params, iterations=nsteps)):
            # Calculate the iteration number based on the current sample
            iteration = i + 1

            # Calculate acceptance fraction for the current step
            acceptance_fraction = np.mean(sampler.acceptance_fraction)

            # Manually control the output to include acceptance fraction
            if (iteration % 1 == 0 or iteration == nsteps):
                # Print out the current iteration, total iterations, and acceptance fraction
                print(f"Step: {iteration}/{nsteps}: acceptance fraction = {acceptance_fraction:.3f}", end="\r")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run MCMC.")
    parser.add_argument("--config", type=str, help="Config data file name.")
    parser.add_argument("--cov", type=str, default=cov_dir, help="Covariance directory.")
    parser.add_argument("--odir", type=str, default=dir_base, help="Output directory.")
    args = parser.parse_args()
    main(args.config, cov_dir=args.cov, outdir=args.odir)
