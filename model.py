import numpy as np
import sys
import warnings
import utils as ut

sys.path.insert(0, '/home/ag5103/szpack.v2.0/python')
sys.path.insert(0, "../")
#os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, '/home/gill/research/act/szpack.v2.0/python')
sys.path.insert(0, '/home/a/ahincks/gillajay/szpack.v2.0/python')

warnings.filterwarnings('ignore')

import SZpack as SZ 

const_c = 299792458.0 # m / s
const_k_B = 1.380649e-23 # J / K
const_h = 6.626070149999999e-25 # J / GHz

class Filament():

    def __init__(self, theta, fit_dust):
       
        self.fit_dust = fit_dust
        self.initialize(theta=theta)

    def initialize(self, theta):

        if self.fit_dust == 'true':
            self.ra_pix = theta[18]
            self.dec_pix = theta[19]
            self.l0_pix = theta[20]
            self.w0_pix = theta[21]
            self.Dtau = theta[22]
            self.Te = theta[23]
            self.A_D = theta[24]
            self.v_avg = theta[25]
            self.theta_bridge = np.deg2rad(117)

        elif self.fit_dust == 'false':
            self.ra_pix = theta[16]
            self.dec_pix = theta[17]
            self.l0_pix = theta[18]
            self.w0_pix = theta[19]
            self.Dtau = theta[20]
            self.Te = theta[21]
            self.v_avg = theta[22]

            self.theta_bridge = np.deg2rad(117)

    def szmodel(self, 
                frequency, 
                xgrid, 
                ygrid,
                z,
                muo, 
                array):
        
        SZ_params = SZ.parameters()
        SZ_params.betao = 0.001233586736861806 # Sun rel to CMB
        SZ_params.muo = muo
        SZ_params.runmode = 'full'
        SZ_params.T_order = 10
        SZ_params.beta_order = 2
        SZ_params.Dtau = self.Dtau

        SZ_params.betac = np.abs(self.v_avg) * 1000 / const_c

        if self.v_avg < 0:
            SZ_params.muc = -1
        else:
            SZ_params.muc = 1

        # # Set higher order terms to zero
        SZ_params.means_assign_omegas(0, 0, 0)
        SZ_params.means_assign_sigmas(0, 0, 0)
        SZ_params.means_kappa = 0

        # Get frequency and bandpass arrays
        freq_str = ut.get_freq_str(frequency=frequency)
        x_array, freq_array, del_freq = ut.get_freq_bandpass(array=array, freq_str=freq_str)
        bandpass, bandpass_int = ut.get_bandpass(array=array, freq_str=freq_str)

        # Set SZ pack frequency
        SZ_params.set_x_array(x_array[0], x_array[-1], len(x_array))

        SZ_params.Te = self.Te

                # Bridge implementation
        w = ( (xgrid - self.ra_pix)*np.sin(self.theta_bridge) 
            + (ygrid - self.dec_pix)*np.cos(self.theta_bridge) )
        
        l = ( (xgrid - self.ra_pix)*np.cos(self.theta_bridge) 
             - (ygrid - self.dec_pix)*np.sin(self.theta_bridge) )

        # Mesa model
        bridge_shape = 1 / (1 + (l/self.l0_pix)**8 + (w/self.w0_pix)**8)

        # SZ effect signal calculation
        DI = SZ.compute_combo(SZ_params, DI=True) * 10**6. # Jy/sr

        # Bandpass integration
        numerator = np.trapz(y=bandpass * DI,
                                x=freq_array, dx=del_freq)
        
        sz_signal = numerator / bandpass_int

        if self.fit_dust == 'true':    
            freq_ref = 545 
            T_D = 20 
            beta_D = 1.5
        
            x_dust = (const_h * freq_array) * (1 + z) / (const_k_B * T_D)     
            x_ref = (const_h * freq_ref) / (const_k_B * T_D)
            
            term_ref = (np.exp(x_ref) - 1)
            term_dust = (np.exp(x_dust) - 1)
            
            term_power = (freq_array * (1+z) / freq_ref)**(beta_D + 3)

            I_dust = self.A_D * term_power * (term_ref / term_dust) 
            
            numerator_dust = np.trapz(y=bandpass * I_dust,
                                        x=freq_array, dx=del_freq)
            
            dust_signal = numerator_dust / bandpass_int

            # Total model   

            dust_signal *= bridge_shape

            total_model = (bridge_shape * sz_signal) + dust_signal

        
        elif self.fit_dust == 'false':
            total_model = bridge_shape * sz_signal

        else:
            raise ValueError(f"fit_dust must be 'true' or 'false'")

        return total_model

class Cluster():
    """
    Elliptical beta profile for cluster
    Temperature profile
    Dust emission (Modified Blackbody)
    """

    def __init__(self, theta, name, fit_dust): 
        self.name = name
        self.fit_dust = fit_dust
        self.initialize(theta=theta)

    def szmodel(self, 
              frequency, 
              z,
              array,
              xgrid, 
              ygrid,
              muo,
              temp_model_type):
        
        # Setup SZpack parameters class
        SZ_params = SZ.parameters()
        SZ_params.betao = 0.001233586736861806
        SZ_params.muo = muo
        SZ_params.runmode = 'full'
        SZ_params.T_order = 10
        SZ_params.beta_order = 2
        SZ_params.Dtau = self.Dtau

        if self.name == "abell401":
            vc = self.v_avg + (520 / 2)
        elif self.name == "abell399":
            vc = self.v_avg - (520 / 2)
        
        SZ_params.betac = np.abs(vc) * 1000 / const_c

        if vc < 0:
            SZ_params.muc = -1
        else:
            SZ_params.muc = 1
        
        # # Set higher order terms to zero
        SZ_params.means_assign_omegas(0, 0, 0)
        SZ_params.means_assign_sigmas(0, 0, 0)
        SZ_params.means_kappa = 0

        # Get frequency and bandpass arrays
        freq_str = ut.get_freq_str(frequency=frequency)
        x_array, freq_array, del_freq = ut.get_freq_bandpass(array=array, freq_str=freq_str)
        bandpass, bandpass_int = ut.get_bandpass(array=array, freq_str=freq_str)

        # Set SZ pack frequency
        SZ_params.set_x_array(x_array[0], x_array[-1], len(x_array))

        # Grid setup
        xprime = ((xgrid - (self.ra_pix))*np.cos(self.theta_cluster) -
                  (ygrid - (self.dec_pix))*np.sin(self.theta_cluster))
        yprime = ((xgrid - (self.ra_pix))*np.sin(self.theta_cluster) +
                  (ygrid - (self.dec_pix))*np.cos(self.theta_cluster))

        r = np.sqrt(xprime**2. + (yprime * self.R)**2.)

        rc_pix = self.rc_arcmin / 0.5

        # Electron density projected map
        beta_density_map_2d = (1 + (r / rc_pix)**2.)**(-1.5*self.beta + 0.5)

        # Interpolation function of SZ signal

        if temp_model_type == 'isothermal':
            
            SZ_params.Te = self.T_e

            I = SZ.compute_combo(SZ_params, DI=True) * 10**6. # Jy/sr

            numerator = np.trapz(y=bandpass * np.array(I),
                                 x=freq_array, dx=del_freq)

            sz_signal = numerator / (bandpass_int) # SZ model
            
        else:
            raise ValueError(f"temp_model_type must be 'isothermal', or 'vikhilin'")

        # Dust emission treatment    
        if self.fit_dust == 'true':    
            freq_ref = 545 
            T_D = 20 
            beta_D = 1.5
        
            x_dust = (const_h * freq_array) * (1 + z) / (const_k_B * T_D)     
            x_ref = (const_h * freq_ref) / (const_k_B * T_D)
            
            term_ref = (np.exp(x_ref) - 1)
            term_dust = (np.exp(x_dust) - 1)
            
            term_power = (freq_array * (1+z) / freq_ref)**(beta_D + 3)

            I_dust = self.A_D * term_power * (term_ref / term_dust) 
            
            numerator_dust = np.trapz(y=bandpass * I_dust,
                                        x=freq_array, dx=del_freq)
            
            dust_signal = numerator_dust / bandpass_int

            # Total model   

            dust_signal *= beta_density_map_2d

            total_model = (beta_density_map_2d * sz_signal) + dust_signal
        
        elif self.fit_dust == 'false':
            total_model = beta_density_map_2d * sz_signal

        else:
            raise ValueError(f"fit_dust must be 'true' or 'false'")

        return total_model

    def initialize(self, theta):

        if self.fit_dust == 'true':
                
            if self.name == "abell401":
                self.ra_pix = theta[0]
                self.dec_pix = theta[1]
                self.beta = theta[2]
                self.rc_arcmin = theta[3]
                self.R = theta[4]
                self.theta_cluster = np.deg2rad(theta[5])
                self.Dtau = theta[6]
                self.T_e = theta[7]
                self.A_D = theta[8]

                self.v_avg = theta[25]
                #self.v_delta = theta[26]

            elif self.name == "abell399":
                self.ra_pix = theta[9]
                self.dec_pix = theta[10]
                self.beta = theta[11]
                self.rc_arcmin = theta[12]
                self.R = theta[13]
                self.theta_cluster = np.deg2rad(theta[14])
                self.Dtau = theta[15]
                self.T_e = theta[16]
                self.A_D = theta[17]

                self.v_avg = theta[25]
                #self.v_delta = theta[26]

        elif self.fit_dust == 'false':
            
            if self.name == "abell401":
                self.ra_pix = theta[0]
                self.dec_pix = theta[1]
                self.beta = theta[2]
                self.rc_arcmin = theta[3]
                self.R = theta[4]
                self.theta_cluster = np.deg2rad(theta[5])
                self.Dtau = theta[6]
                self.T_e = theta[7]

                self.v_avg = theta[22]
                #self.v_delta = theta[23]

            elif self.name == "abell399":
                self.ra_pix = theta[8]
                self.dec_pix = theta[9]
                self.beta = theta[10]
                self.rc_arcmin = theta[11]
                self.R = theta[12]
                self.theta_cluster = np.deg2rad(theta[13])
                self.Dtau = theta[14]
                self.T_e = theta[15]

                self.v_avg = theta[22]
                #self.v_delta = theta[23]

    
def vikhlinin_temp_profile(T_e, x_prof):
    T_r_num = 1.35 * (T_e / 1.11) * ( (x_prof / 0.045)**(1.9) + 0.45 ) 
    T_r_den = ( ((x_prof / 0.045)**(1.9) + 1) * (1 + (x_prof / 0.6)**2)**(0.45) )
    T_r = T_r_num / T_r_den
    return T_r
    
# def init_SZ_class():
#     SZ_params = SZ.parameters()
#     SZ_params.betao = 0.001241
#     SZ_params.muo = 0
#     SZ_params.muc = 1
#     SZ_params.runmode = 'full'
#     SZ_params.T_order = 10
#     SZ_params.beta_order = 2
#     return SZ_params

def range_check(array, min, max, use_min=True, use_max=True):
    range_fail = False
    if use_min:
        if np.min(array) < min: 
            range_fail = True
    if use_max:
        if np.max(array) > max:
            range_fail = True
    return range_fail

# class Cluster():
#     """
#     Elliptical beta profile for cluster
#     Temperature profile
#     Dust emission (Modified Blackbody)
#     """

#     def __init__(self, theta, name, fit_type, ra_pix, dec_pix): 
#         self.name = name
#         self.fit_type = fit_type
        
#         self.initialize(theta=theta, ra_pix=ra_pix, dec_pix=dec_pix)

#     def szmodel(self, 
#               frequency, 
#               z,
#               array,
#               inst,
#               scan,
#               xgrid, 
#               ygrid,
#               interp_f_dict,
#               temp_model_type,
#               r500_pix):
        
#         # Setup SZpack parameters class
#         SZ_params = init_SZ_class()

#         # Get frequency and bandpass arrays
#         freq_str = ut.get_freq_str(frequency=frequency)
#         x_array, freq_array, del_freq = ut.get_freq_bandpass(array=array, freq_str=freq_str)
#         bandpass, bandpass_int = ut.get_bandpass(array=array, freq_str=freq_str)
        
#         # Set SZ pack frequency
#         SZ_params.set_x_array(x_array[0], x_array[-1], len(x_array))

#         # Grid setup
#         xprime = ((xgrid - (self.ra_pix))*np.cos(self.theta_cluster) -
#                   (ygrid - (self.dec_pix))*np.sin(self.theta_cluster))
#         yprime = ((xgrid - (self.ra_pix))*np.sin(self.theta_cluster) +
#                   (ygrid - (self.dec_pix))*np.cos(self.theta_cluster))

#         r = np.sqrt(xprime**2. + (yprime * self.R)**2.)

#         rc_pix = self.rc_arcmin / 0.5

#         # Interpolation function of SZ signal
#         interp_f = interp_f_dict[f"{int(frequency)}_{array}"]

#         # Electron density projected map
#         beta_density_map_2d = (1 + (r / rc_pix)**2.)**(-1.5*self.beta + 0.5)

#         # # Simple model
#         # SZ_params.Te = self.T_e
#         # SZ_params.Dtau = self.Dtau
#         # I = SZ.compute_combo(SZ_params, DI=True) * 10**6. # Jy/sr

#         # numerator = np.trapz(y=bandpass * I,
#         #                      x=freq_array, dx=del_freq)
        
#         # sz_signal = numerator / (bandpass_int) # SZ model

#         # Optical depth projected map
#         tau_map_2d = self.Dtau * beta_density_map_2d
        
#         if temp_model_type == 'isothermal':
            
#             SZ_params.Te = self.T_e
            
#             vc_range_check = range_check(array=self.vc, min=-10000, max=10000)
#             Te_range_check = range_check(array=self.T_e, min=0, max=20)
#             tau_range_check = range_check(array=tau_map_2d, min=0, max=0.1)

#             if vc_range_check or Te_range_check or tau_range_check:
#                 SZ_map = np.zeros_like(tau_map_2d)
            
#             else:
#                 SZ_map = interp_f((self.vc, self.T_e, tau_map_2d))

#         elif temp_model_type == 'vikhilin':
            
#             x_prof = r / r500_pix
            
#             T_r = vikhlinin_temp_profile(T_e=self.T_e, x_prof=x_prof)

#             vc_range_check = range_check(array=self.vc, min=-10000, max=10000)
#             Te_range_check = range_check(array=T_r, min=0, max=20)
#             tau_range_check = range_check(array=tau_map_2d, min=0, max=0.1)

#             if vc_range_check or Te_range_check or tau_range_check:
#                 SZ_map = np.zeros_like(tau_map_2d)
            
#             else:
#                 SZ_map = interp_f((self.vc, T_r, tau_map_2d))

#         else:
#             raise ValueError(f"temp_model_type must be 'isothermal', or 'vikhilin'")

#         # Dust emission treatment        
#         if "dust" in self.fit_type:
#                 freq_ref = 545 * u.GHz
#                 T_D = 20 * u.K
#                 beta_D = 1.5
            
#                 x_dust = (const.h.to(u.J/u.GHz) * freq_array*u.GHz) * (1 + z) / (const.k_B * T_D)     
#                 x_ref = (const.h.to(u.J/u.GHz) * freq_ref) / (const.k_B * T_D)
                
#                 term_ref = (np.exp(x_ref) - 1)
#                 term_dust = (np.exp(x_dust) - 1)
                
#                 term_power = (freq_array*u.GHz * (1+z) / freq_ref)**(beta_D + 3)
                
#                 I_dust = ( (self.A_D * u.Jansky/u.steradian).value 
#                         * term_power * (term_ref / term_dust) )
                
#                 numerator_dust = np.trapz(y=bandpass * I_dust,
#                                           x=freq_array, dx=del_freq)
                
#                 dust_signal = numerator_dust / bandpass_int
#         else: 
#             dust_signal = 0
        
#         # Background fit
#         if "bkg" in self.fit_type:
#             bkg_level = self.bkg_dict[f"bkg_A_{(freq_str)}_{array}_{inst}_{scan}"]

#             total_model = SZ_map + bkg_level + (beta_density_map_2d * dust_signal)
        
#         else:
#             total_model = SZ_map + (beta_density_map_2d * dust_signal)
    
#         return total_model
    
    # def initialize(self, theta, ra_pix, dec_pix):

    #     if self.fit_type == "pos_dust_bkg":
    #         self.ra_pix = theta[0]
    #         self.dec_pix = theta[1]
    #         self.beta = theta[2]
    #         self.rc_arcmin = theta[3]
    #         self.R = theta[4]
    #         self.theta_cluster = np.deg2rad(theta[5])
    #         self.Dtau = theta[6]
    #         self.T_e = theta[7]
    #         self.vc = theta[8]
    #         self.A_D = theta[9]

    #         self.bkg_A_150_pa4_act_night = theta[10]
    #         self.bkg_A_220_pa4_act_night = theta[11]
    #         self.bkg_A_150_pa5_act_night = theta[12]
    #         self.bkg_A_098_pa5_act_night = theta[13]
    #         self.bkg_A_098_pa6_act_night = theta[14]
    #         self.bkg_A_150_pa6_act_night = theta[15]

    #         self.bkg_A_030_npipe_planck_sky = theta[16]
    #         self.bkg_A_044_npipe_planck_sky = theta[17]
    #         self.bkg_A_070_npipe_planck_sky = theta[18]

    #         self.bkg_A_100_npipe_planck_sky = theta[19]
    #         self.bkg_A_143_npipe_planck_sky = theta[20]

    #         self.bkg_A_217_npipe_planck_sky = theta[21]
    #         self.bkg_A_353_npipe_planck_sky = theta[22]
    #         self.bkg_A_545_npipe_planck_sky = theta[23]
    #         self.bkg_A_857_npipe_planck_sky = theta[24]


    #         self.bkg_dict = {"bkg_A_150_pa4_act_night": self.bkg_A_150_pa4_act_night,
    #             "bkg_A_220_pa4_act_night": self.bkg_A_220_pa4_act_night,
    #             "bkg_A_150_pa5_act_night": self.bkg_A_150_pa5_act_night,
    #             "bkg_A_098_pa5_act_night": self.bkg_A_098_pa5_act_night,
    #             "bkg_A_098_pa6_act_night": self.bkg_A_098_pa6_act_night,
    #             "bkg_A_150_pa6_act_night": self.bkg_A_150_pa6_act_night,

    #             "bkg_A_030_npipe_planck_sky": self.bkg_A_030_npipe_planck_sky,
    #             "bkg_A_044_npipe_planck_sky": self.bkg_A_044_npipe_planck_sky,
    #             "bkg_A_070_npipe_planck_sky": self.bkg_A_070_npipe_planck_sky,

    #             "bkg_A_100_npipe_planck_sky": self.bkg_A_100_npipe_planck_sky,
    #             "bkg_A_143_npipe_planck_sky": self.bkg_A_143_npipe_planck_sky,

    #             "bkg_A_217_npipe_planck_sky": self.bkg_A_217_npipe_planck_sky,
    #             "bkg_A_353_npipe_planck_sky": self.bkg_A_353_npipe_planck_sky,
    #             "bkg_A_545_npipe_planck_sky": self.bkg_A_545_npipe_planck_sky,
    #             "bkg_A_857_npipe_planck_sky": self.bkg_A_857_npipe_planck_sky
    #             }
            
    #     elif self.fit_type == "pos_dust":
    #         self.ra_pix = theta[0]
    #         self.dec_pix = theta[1]
    #         self.beta = theta[2]
    #         self.rc_arcmin = theta[3]
    #         self.R = theta[4]
    #         self.theta_cluster = np.deg2rad(theta[5])
    #         self.Dtau = theta[6]
    #         self.T_e = theta[7]
    #         self.vc = theta[8]
    #         self.A_D = theta[9]

    #     elif self.fit_type == "pos_bkg":
    #         self.ra_pix = theta[0]
    #         self.dec_pix = theta[1]
    #         self.beta = theta[2]
    #         self.rc_arcmin = theta[3]
    #         self.R = theta[4]
    #         self.theta_cluster = np.deg2rad(theta[5])
    #         self.Dtau = theta[6]
    #         self.T_e = theta[7]
    #         self.vc = theta[8]

    #         self.bkg_A_150_pa4_act_night = theta[9]
    #         self.bkg_A_220_pa4_act_night = theta[10]
    #         self.bkg_A_150_pa5_act_night = theta[11]
    #         self.bkg_A_098_pa5_act_night = theta[12]
    #         self.bkg_A_098_pa6_act_night = theta[13]
    #         self.bkg_A_150_pa6_act_night = theta[14]

    #         self.bkg_A_030_npipe_planck_sky = theta[15]
    #         self.bkg_A_044_npipe_planck_sky = theta[16]
    #         self.bkg_A_070_npipe_planck_sky = theta[17]
    #         self.bkg_A_100_npipe_planck_sky = theta[18]
    #         self.bkg_A_143_npipe_planck_sky = theta[19]
    #         self.bkg_A_217_npipe_planck_sky = theta[20]
    #         self.bkg_A_353_npipe_planck_sky = theta[21]
    #         self.bkg_A_545_npipe_planck_sky = theta[22]
    #         self.bkg_A_857_npipe_planck_sky = theta[23]

    #         self.bkg_dict = {"bkg_A_150_pa4_act_night": self.bkg_A_150_pa4_act_night,
    #             "bkg_A_220_pa4_act_night": self.bkg_A_220_pa4_act_night,
    #             "bkg_A_150_pa5_act_night": self.bkg_A_150_pa5_act_night,
    #             "bkg_A_098_pa5_act_night": self.bkg_A_098_pa5_act_night,
    #             "bkg_A_098_pa6_act_night": self.bkg_A_098_pa6_act_night,
    #             "bkg_A_150_pa6_act_night": self.bkg_A_150_pa6_act_night,

    #             "bkg_A_030_npipe_planck_sky": self.bkg_A_030_npipe_planck_sky,
    #             "bkg_A_044_npipe_planck_sky": self.bkg_A_044_npipe_planck_sky,
    #             "bkg_A_070_npipe_planck_sky": self.bkg_A_070_npipe_planck_sky,
    #             "bkg_A_100_npipe_planck_sky": self.bkg_A_100_npipe_planck_sky,
    #             "bkg_A_143_npipe_planck_sky": self.bkg_A_143_npipe_planck_sky,
    #             "bkg_A_217_npipe_planck_sky": self.bkg_A_217_npipe_planck_sky,
    #             "bkg_A_353_npipe_planck_sky": self.bkg_A_353_npipe_planck_sky,
    #             "bkg_A_545_npipe_planck_sky": self.bkg_A_545_npipe_planck_sky,
    #             "bkg_A_857_npipe_planck_sky": self.bkg_A_857_npipe_planck_sky
    #             }

    #     elif self.fit_type == "dust_bkg":
    #         self.ra_pix = ra_pix
    #         self.dec_pix = dec_pix

    #         self.beta = theta[0]
    #         self.rc_arcmin = theta[1]
    #         self.R = theta[2]
    #         self.theta_cluster = np.deg2rad(theta[3])
    #         self.Dtau = theta[4]
    #         self.T_e = theta[5]
    #         self.vc = theta[6]
    #         self.A_D = theta[7]

    #         self.bkg_A_150_pa4_act_night = theta[8]
    #         self.bkg_A_220_pa4_act_night = theta[9]
    #         self.bkg_A_150_pa5_act_night = theta[10]
    #         self.bkg_A_098_pa5_act_night = theta[11]
    #         self.bkg_A_098_pa6_act_night = theta[12]
    #         self.bkg_A_150_pa6_act_night = theta[13]

    #         self.bkg_A_030_npipe_planck_sky = theta[14]
    #         self.bkg_A_044_npipe_planck_sky = theta[15]
    #         self.bkg_A_070_npipe_planck_sky = theta[16]
    #         self.bkg_A_100_npipe_planck_sky = theta[17]
    #         self.bkg_A_143_npipe_planck_sky = theta[18]

    #         self.bkg_A_217_npipe_planck_sky = theta[19]
    #         self.bkg_A_353_npipe_planck_sky = theta[20]
    #         self.bkg_A_545_npipe_planck_sky = theta[21]
    #         self.bkg_A_857_npipe_planck_sky = theta[22]
     
    #         self.bkg_dict = {"bkg_A_150_pa4_act_night": self.bkg_A_150_pa4_act_night,
    #             "bkg_A_220_pa4_act_night": self.bkg_A_220_pa4_act_night,
    #             "bkg_A_150_pa5_act_night": self.bkg_A_150_pa5_act_night,
    #             "bkg_A_098_pa5_act_night": self.bkg_A_098_pa5_act_night,
    #             "bkg_A_098_pa6_act_night": self.bkg_A_098_pa6_act_night,
    #             "bkg_A_150_pa6_act_night": self.bkg_A_150_pa6_act_night,

    #             "bkg_A_030_npipe_planck_sky": self.bkg_A_030_npipe_planck_sky,
    #             "bkg_A_044_npipe_planck_sky": self.bkg_A_044_npipe_planck_sky,
    #             "bkg_A_070_npipe_planck_sky": self.bkg_A_070_npipe_planck_sky,
    #             "bkg_A_100_npipe_planck_sky": self.bkg_A_100_npipe_planck_sky,
    #             "bkg_A_143_npipe_planck_sky": self.bkg_A_143_npipe_planck_sky,
    #             "bkg_A_217_npipe_planck_sky": self.bkg_A_217_npipe_planck_sky,
    #             "bkg_A_353_npipe_planck_sky": self.bkg_A_353_npipe_planck_sky,
    #             "bkg_A_545_npipe_planck_sky": self.bkg_A_545_npipe_planck_sky,
    #             "bkg_A_857_npipe_planck_sky": self.bkg_A_857_npipe_planck_sky,
    #             }   

    #     elif self.fit_type == "pos":
    #         self.ra_pix = theta[0]
    #         self.dec_pix = theta[1]
    #         self.beta = theta[2]
    #         self.rc_arcmin = theta[3]
    #         self.R = theta[4]
    #         self.theta_cluster = np.deg2rad(theta[5])
    #         self.Dtau = theta[6]
    #         self.T_e = theta[7]
    #         self.vc = theta[8]

    #     elif self.fit_type == "dust":
    #         self.ra_pix = ra_pix
    #         self.dec_pix = dec_pix

    #         self.beta = theta[0]
    #         self.rc_arcmin = theta[1]
    #         self.R = theta[2]
    #         self.theta_cluster = np.deg2rad(theta[3])
    #         self.Dtau = theta[4]
    #         self.T_e = theta[5]
    #         self.vc = theta[6]
    #         self.A_D = theta[7]

    #     elif self.fit_type == "bkg":
    #         self.ra_pix = ra_pix
    #         self.dec_pix = dec_pix

    #         self.beta = theta[0]
    #         self.rc_arcmin = theta[1]
    #         self.R = theta[2]
    #         self.theta_cluster = np.deg2rad(theta[3])
    #         self.Dtau = theta[4]
    #         self.T_e = theta[5]
    #         self.vc = theta[6]

    #         self.bkg_A_150_pa4_act_night = theta[7]
    #         self.bkg_A_220_pa4_act_night = theta[8]
    #         self.bkg_A_150_pa5_act_night = theta[9]
    #         self.bkg_A_098_pa5_act_night = theta[10]
    #         self.bkg_A_098_pa6_act_night = theta[11]
    #         self.bkg_A_150_pa6_act_night = theta[12]

    #         self.bkg_A_030_npipe_planck_sky = theta[13]
    #         self.bkg_A_044_npipe_planck_sky = theta[14] 
    #         self.bkg_A_070_npipe_planck_sky = theta[15]
    #         self.bkg_A_100_npipe_planck_sky = theta[16]
    #         self.bkg_A_143_npipe_planck_sky = theta[17]
    #         self.bkg_A_217_npipe_planck_sky = theta[18]
    #         self.bkg_A_353_npipe_planck_sky = theta[19]
    #         self.bkg_A_545_npipe_planck_sky = theta[20]
    #         self.bkg_A_857_npipe_planck_sky = theta[21]

    #         self.bkg_dict = {"bkg_A_150_pa4_act_night": self.bkg_A_150_pa4_act_night,
    #                          "bkg_A_220_pa4_act_night": self.bkg_A_220_pa4_act_night,
    #                          "bkg_A_150_pa5_act_night": self.bkg_A_150_pa5_act_night,
    #                          "bkg_A_098_pa5_act_night": self.bkg_A_098_pa5_act_night,
    #                          "bkg_A_098_pa6_act_night": self.bkg_A_098_pa6_act_night,
    #                          "bkg_A_150_pa6_act_night": self.bkg_A_150_pa6_act_night, 
    #                          "bkg_A_030_npipe_planck_sky": self.bkg_A_030_npipe_planck_sky,
    #                          "bkg_A_044_npipe_planck_sky": self.bkg_A_044_npipe_planck_sky,
    #                          "bkg_A_070_npipe_planck_sky": self.bkg_A_070_npipe_planck_sky,
    #                          "bkg_A_100_npipe_planck_sky": self.bkg_A_100_npipe_planck_sky,
    #                          "bkg_A_143_npipe_planck_sky": self.bkg_A_143_npipe_planck_sky,
    #                          "bkg_A_217_npipe_planck_sky": self.bkg_A_217_npipe_planck_sky,
    #                          "bkg_A_353_npipe_planck_sky": self.bkg_A_353_npipe_planck_sky,
    #                          "bkg_A_545_npipe_planck_sky": self.bkg_A_545_npipe_planck_sky,
    #                          "bkg_A_857_npipe_planck_sky": self.bkg_A_857_npipe_planck_sky
    #                          }
            
    #     else: 
    #         raise ValueError("fit_type must be 'pos_dust_bkg', 'pos_dust', \
    #                          'pos_bkg', 'dust_bkg', 'pos', 'dust', or 'bkg'")

