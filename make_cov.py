# %%
import dotenv
dotenv.load_dotenv()

import sys, os
dir_head = os.environ.get("REPO_DIR", ".")
sys.path.insert(0, os.path.join(dir_head, "src"))

import numpy as np
from pixell import enmap

import warnings
import itertools

warnings.filterwarnings("ignore")

# files from the src dir in git
import utils as ut
import covariance as cov

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--save-cov", action="store_true")
parser.add_argument("--cov-dir", type=str, default=os.environ.get("COV_DIR", "."))
parser.add_argument("--show-plot", action="store_true")
parser.add_argument("--sm45", help="smooth act pa4/5 cov matrix", action="store_true")
args = parser.parse_args()

save_cov = args.save_cov
cov_dir = args.cov_dir
show_plot = args.show_plot
smooth_cov = args.sm45

# %%
config_data_fname = os.path.join(dir_head, "configs/config_mcmc.yaml")

config_data = ut.get_config_file(config_data_fname)
cluster_region = ut.get_region(region_center_ra=config_data['region_center_ra'], 
                               region_center_dec=config_data['region_center_dec'],
                               region_width=config_data['region_width'])

fname = os.getenv('ACT_DATADIR') + "act_cut_dr6v2_pa6_f150_4way_coadd_map_srcfree.fits"

cluster_map = enmap.read_map(fname=fname, box=cluster_region)
data_shape, data_wcs = cluster_map.shape, cluster_map.wcs

combos = []
for combo in itertools.product(config_data['data'], config_data['data']):
    combos.append(combo)

for idx, combo in enumerate(combos):
    freq1 = combo[0].split('_')[0]
    array1 = combo[0].split('_')[1]
    inst1 = combo[0].split('_')[2]
    scan1 = combo[0].split('_')[3]

    freq2 = combo[1].split('_')[0]
    array2 = combo[1].split('_')[1]
    inst2 = combo[1].split('_')[2]
    scan2 = combo[1].split('_')[3]

    if inst1 == 'act':
        data_dir1 = os.getenv('ACT_DATADIR')
    elif inst1 == 'planck':
        data_dir1 = os.getenv('PLANCK_DATADIR')
    if inst2 == 'act':
        data_dir2 = os.getenv('ACT_DATADIR')
    elif inst2 == 'planck':
        data_dir2 = os.getenv('PLANCK_DATADIR')

    print("Calculating covariance for: {}".format(combo))

    mean_tpsd, mean_npsd, mean_spsd, all_regions_npsd, all_regions_spsd = cov.get_covariance(
        freq1=freq1, 
        freq2=freq2,
                      
        array1=array1, 
        array2=array2,
                      
        inst1=inst1, 
        inst2=inst2,
                      
        data_dir1=data_dir1,
        data_dir2=data_dir2,
                                                         
        data_wcs=data_wcs,
                                                         
        cf=config_data
    )

    # smooth cov for pa4 and pa5 by fitting a model -> remove the model -> smooth residue -> reintroduce the model
    # only do this when calculating within act cov with pa4 or pa5
    doit = False
    if ('pa4' in [array1, array2]) or ('pa5' in [array1, array2]):
        doit = True
    if 'planck' in [inst1, inst2]:
        doit = False
    if smooth_cov and doit:
        print("Smoothing cov for pa4 and pa5: array1 = {}, array2 = {}".format(array1, array2))
        if 'pa4' in [array1, array2] and 'pa5' in [array1, array2]:
            npsd_orig = all_regions_npsd[-1]
        npsd_orig = all_regions_npsd[-1]  # center region
        b_npsd, l_npsd = enmap.lbin(map=np.abs(npsd_orig), bsize=200)
        geometry = (data_shape, data_wcs) 
        npsd_smooth = cov.smoothing(l_npsd, b_npsd, npsd_orig, 
                                    config_data['smooth_noise_pix'], 
                                    config_data['min_ell_fit'], geometry)
        # recalculate total cov = signal + noise = mean_signal + smoothed_noise
        mean_tpsd = mean_spsd + npsd_smooth

    if show_plot:
        import matplotlib.pyplot as plt

        b_tpsd, l_tpsd = enmap.lbin(map=np.abs(mean_tpsd), bsize=200)
        b_npsd, l_npsd = enmap.lbin(map=np.abs(mean_npsd), bsize=200)
        b_spsd, l_spsd = enmap.lbin(map=np.abs(mean_spsd), bsize=200)

        plt.loglog(l_tpsd, b_tpsd, label="Total")
        plt.loglog(l_spsd, b_spsd, label="Signal")
        plt.loglog(l_npsd, b_npsd, label="Noise")
        plt.legend()
        plt.grid()
        plt.show()

        ut.plot_image(np.fft.fftshift(np.abs(mean_tpsd) - np.abs(mean_spsd)), interval_type="zscale")

    if save_cov:
        np.save(file=f"{cov_dir}/cov_{freq1}_{array1}_{inst1}_{scan1}_{freq2}_{array2}_{inst2}_{scan2}.npy", arr=mean_tpsd)

