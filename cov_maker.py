# %%
# Env
import dotenv
dotenv.load_dotenv()

import sys, os
dir_head = os.environ.get("REPO_DIR", ".")
sys.path.insert(0, os.path.join(dir_head, "src"))

import numpy as np
from pixell import enmap

import warnings
import itertools
from scipy.ndimage import gaussian_filter
import multiprocessing
from functools import partial

warnings.filterwarnings("ignore")

# Files from the src dir in git
import utils as ut
import covariance as cov

import matplotlib.pyplot as plt

#%%


# %%
config_data_fname = os.environ["REPO_DIR"] + "/configs/config_mcmc.yaml"

config_data = ut.get_config_file(config_data_fname)

import random
# Dictionary to store scaling factors for each unique pa4-pa5 pair
combos = []
for combo in itertools.product(config_data['data'], config_data['data']):
    combos.append(combo)
scale_factors = {}
# Iterate through the list of pairs
# for pair in combos:
#     sorted_pair = tuple(sorted(pair))
#     # Check if the pair involves pa4 and pa5
#     if ('pa4' in pair[0] and 'pa5' in pair[1]) or ('pa4' in pair[1] and 'pa5' in pair[0]):
#         # Create a unique identifier for the pair regardless of order
#         # If this pair hasn't been scaled yet, generate a scale factor
#         if sorted_pair not in scale_factors:
#             scale_factors[sorted_pair] = random.uniform(0.5, 1.5)  # Example scaling factor range
#         # Apply the same scale factor for reverse order pairs
#         # scale_factor = scale_factors[sorted_pair]
#         # print(f"Pair: {pair}, Scale Factor: {scale_factor}")
#     else:
#         scale_factors[sorted_pair] = 1

# print(scale_factors)

#%%
cluster_region = ut.get_region(region_center_ra=config_data['region_center_ra'], 
                               region_center_dec=config_data['region_center_dec'],
                               region_width=config_data['region_width'])

fname = os.getenv("ACT_DATADIR") + "act_cut_dr6v2_pa6_f150_4way_coadd_map_srcfree.fits"

save_cov = 1
plot_1d = 0
plot_2d = 0

data_ref = enmap.read_map(fname=fname, box=cluster_region)
data_shape, data_wcs = data_ref.shape, data_ref.wcs

if save_cov: 
    os.system(f"rm -rf {os.environ['COV_DIR']}/cov_*npy*")



def process_combo_cov(combo, 
                      geometry=(data_shape, data_wcs), 
                      plot_1d=False, 
                      plot_2d=False, 
                      save_cov=True,
                      config_data=config_data):
    freq1 = combo[0].split('_')[0]
    array1 = combo[0].split('_')[1]
    inst1 = combo[0].split('_')[2]
    scan1 = combo[0].split('_')[3]

    freq2 = combo[1].split('_')[0]
    array2 = combo[1].split('_')[1]
    inst2 = combo[1].split('_')[2]
    scan2 = combo[1].split('_')[3]

    if inst1 == 'act':
        data_dir1 = os.getenv("ACT_DATADIR")
    elif inst1 == 'planck':
        data_dir1 = os.getenv("PLANCK_DATADIR")
    
    if inst2 == 'act':
        data_dir2 = os.getenv("ACT_DATADIR")
    elif inst2 == 'planck':
        data_dir2 = os.getenv("PLANCK_DATADIR")

    # print("Calculating covariance for: {}".format(combo))
    if 0:
        debug_pix_noise = (inst1 == 'planck', inst2 == 'planck')
    else:
        debug_pix_noise = (False, False)

    mean_tpsd, mean_npsd, mean_spsd, all_regions_npsd, all_regions_spsd = cov.get_covariance(freq1=freq1, 
                                                                                    freq2=freq2,
                                                
                                                                                    array1=array1, 
                                                                                    array2=array2,
                                                
                                                                                    inst1=inst1, 
                                                                                    inst2=inst2,
                                                
                                                                                    data_dir1=data_dir1,
                                                                                    data_dir2=data_dir2,
                                                                                    
                                                                                    data_wcs=geometry[1],
                                                                                    
                                                                                    cf=config_data,
                                                                                    debug_pix_noise=debug_pix_noise,)

    # rescale planck noise level below pixel scale such that ACT will dominate in those scales
    # if (inst1 == 'planck' and inst2 == 'planck'):
    #     print(f"combo = {combo}")
    #     modlmap = enmap.modlmap(data_shape, data_wcs)
    #     m = np.logical_and(modlmap >= 5000, modlmap <= 6000)
    #     mean_var = np.mean(mean_tpsd[m]).real * 100
    #     # mean_var = np.inf
    #     # mean_var = 1e25
    #     print(f"Mean variance: {mean_var:.2e}")
    #     m2 = modlmap>6000
    #     mean_tpsd[m2] = mean_var
    #     mean_tpsd = mean_tpsd + 0j

    # Do the scaling under these cases
    if 0:
    # if (inst1 == 'act' and inst2 == 'act') and (array1 in ['pa4', 'pa5'] or array2 in ['pa4', 'pa5']):
    # if ( (inst1 == 'act' and inst2 == 'act') 
    #      and (array1 in ['pa4', 'pa5'] 
    #      or array2 in ['pa4', 'pa5'])
    #      and (array1 == array2) ):

        # print("Scaling the noise power spectrum")
        
        binsize = 200

        # get the avg noise power spectrum in 1D space from all the regions
        # b_regions, l_regions = [], []
        # for region in all_regions_npsd:
        #     b_npsd, l_npsd = enmap.lbin(map=np.abs(region), bsize=binsize)
        #     b_regions.append(b_npsd) 
        #     l_regions.append(l_npsd)

        # b_avg = np.mean(b_regions[:-1], axis=0)
        # l_avg = np.mean(l_regions[:-1], axis=0)
        mean_npsd_regions = enmap.ndmap(np.mean(all_regions_npsd[:-1], axis=0), wcs=geometry[1])
        b_avg, l_avg = enmap.lbin(map=np.abs(mean_npsd_regions), bsize=binsize)

        # npsd from center region
        # b_npsd_center, l_npsd_center = b_regions[-1], l_regions[-1]
        b_npsd_center, l_npsd_center = enmap.lbin(map=np.abs(all_regions_npsd[-1]), bsize=binsize)

        # 2D fit over all regions except the center
        npsd_2d_regions_fit = cov.fit_one_over_f(l_npsd=l_avg,
                                            b_npsd=b_avg,
                                            # twoD_npsd_orig=np.mean(all_regions_npsd[:-1], axis=0),
                                            # gauss_smooth_sigma=config_data["smooth_noise_pix"],
                                            mask_value=config_data["min_ell_fit"],
                                            geometry=geometry)
        
        # 2D fit over the center region
        npsd_center_fit = cov.fit_one_over_f(l_npsd=l_npsd_center,
                                        b_npsd=b_npsd_center,
                                        # twoD_npsd_orig=all_regions_npsd[-1],
                                        # gauss_smooth_sigma=config_data["smooth_noise_pix"],
                                        mask_value=config_data["min_ell_fit"],
                                        geometry=geometry)
        
        scale_factor = npsd_center_fit / npsd_2d_regions_fit
        modlmap = enmap.modlmap(*geometry) 

        scale_factor[0][0] = 1
        scale_factor[modlmap < config_data["min_ell_fit"]] = 1
        # scale_factor = np.median(scale_factor[modlmap > 3000])
        # scale_factors[tuple(combo)] = scale_factor
        # scale_factor = np.round(scale_factor, 4) * 4
        # scale_factor = 0.7 
        # scale only autocase
        if combo[0] != combo[1]:
            scale_factor = 1
        print(f"{tuple(combo)}: {scale_factor}")

        # ut.plot_image(image=np.fft.fftshift(scale_factor),
        #               title="Scale factor",
        #               stretch='log',
        #               interval_type='simple_norm')

        # scale_factor = 1
        # scale_factor = scale_factors[tuple(sorted(combo))]
        # print(f"Scale factor: {scale_factor}")
        # print(f"combo: {combo}")
        # scale_factor = np.mean(scale_factor[modlmap > 5000])
        # print(f"Scale factor: {scale_factor}")
        # combo_seed = hash(tuple(sorted(combo))) % 1000
        # np.random.seed(combo_seed)
        # scale_factor = np.random.uniform(0.5, 1.5)
        # scale_factor = round(scale_factor, 4)
        # print(f"{combo}: {scale_factor}")


        # corrected 2D noise power spectrum
        mean_npsd = np.mean(all_regions_npsd[:-1], axis=0) * scale_factor
        # mean_npsd = np.mean(all_regions_npsd, axis=0) * scale_factor
        mean_npsd = enmap.ndmap(np.array(mean_npsd), wcs=data_wcs)

        mean_tpsd = mean_spsd + mean_npsd

        # b_scaled, l_scaled = enmap.lbin(map=np.abs(mean_npsd), bsize=binsize)
        # b_center_fit, l_center_fit = enmap.lbin(map=np.abs(npsd_center_fit), bsize=binsize)
        # b_regions_fit, l_regions_fit = enmap.lbin(map=np.abs(npsd_2d_regions_fit), bsize=binsize)

        # plt.loglog(l_avg, b_avg, label="Average noise power spectrum")
        # plt.loglog(l_npsd_center, b_npsd_center, label="Center noise power spectrum")
        # plt.loglog(l_scaled, b_scaled, label="Scaled noise power spectrum")
        # # plt.loglog(l_center_fit, b_center_fit, label="Center noise power spectrum fit")
        # # plt.loglog(l_regions_fit, b_regions_fit, label="Average noise power spectrum fit")
        # plt.legend()
        # plt.grid()
        # plt.show()

        if config_data['smooth_total']:
            mean_tpsd = gaussian_filter(input=mean_tpsd, sigma=config_data['smooth_total_pix'])
            mean_tpsd = enmap.ndmap(np.array(mean_tpsd), wcs=data_wcs)

    if plot_1d:
        b_npsd, l_npsd = enmap.lbin(map=np.abs(mean_npsd), bsize=binsize)
        b_tpsd, l_tpsd = enmap.lbin(map=np.abs(mean_tpsd), bsize=binsize)
        b_spsd, l_spsd = enmap.lbin(map=np.abs(mean_spsd), bsize=binsize)

        plt.figure(figsize=(7, 3))
        plt.loglog(l_npsd, b_npsd, label="Noise")
        plt.loglog(l_tpsd, b_tpsd, label="Total")
        plt.loglog(l_spsd, b_spsd, label="Signal")
        plt.title("Absolute value of the power spectrum", fontsize=10)
        plt.legend(fontsize=8)
        plt.grid()

    if plot_2d:
        ut.plot_image(image=np.fft.fftshift(np.abs(mean_tpsd)), 
                      title="Total power spectrum",
                      interval_type='zscale')

    if save_cov:
        np.save(file=f"{os.environ['COV_DIR']}/cov_{freq1}_{array1}_{inst1}_{scan1}_{freq2}_{array2}_{inst2}_{scan2}.npy", arr=mean_tpsd)


combos = list(itertools.product(config_data['data'], config_data['data']))
num_processes = multiprocessing.cpu_count()
print(f"Number of processes: {num_processes}")

with multiprocessing.Pool(processes=num_processes) as pool:
    pool.map(process_combo_cov, combos)