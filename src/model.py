import numpy as np
import warnings
import utils as ut

warnings.filterwarnings('ignore')

import SZpack as SZ 

const_c = 299792458.0 # m / s
const_k_B = 1.380649e-23 # J / K
const_h = 6.626070149999999e-25 # J / GHz

class Filament():

    def __init__(self, theta):
       
        self.initialize(theta=theta)

    def initialize(self, theta):

        self.ra_pix = theta[18]
        self.dec_pix = theta[19]
        self.l0_pix = theta[20]
        self.w0_pix = theta[21]
        self.Dtau = theta[22]
        self.Te = theta[23]
        self.A_D = theta[24]
        self.v_avg = theta[25]
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

        # dust signal
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

        # modulate dust signal with bridge shape (mesa model)
        total_model = bridge_shape * (sz_signal + dust_signal)

        return total_model

class Cluster():
    """
    Elliptical beta profile for cluster
    Temperature profile
    Dust emission (Modified Blackbody)
    """

    def __init__(self, theta, name): 
        self.name = name
        self.initialize(theta=theta)

    def szmodel(self, 
              frequency, 
              z,
              array,
              xgrid, 
              ygrid,
              muo,
              ellipticity_type):
        
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

        # ajay adding this for debugging
        if ellipticity_type == "numerator":
            r = np.sqrt(xprime**2. + (yprime * self.R)**2.)
        elif ellipticity_type == "denominator":
            r = np.sqrt(xprime**2. + (yprime / self.R)**2.) # bounded between 0 and 1
        else:
            raise ValueError("Invalid ellipticity type.")

        rc_pix = self.rc_arcmin / 0.5

        # Electron density projected map
        beta_density_map_2d = (1 + (r / rc_pix)**2.)**(-1.5*self.beta + 0.5)

        SZ_params.Te = self.T_e

        I = SZ.compute_combo(SZ_params, DI=True) * 10**6. # Jy/sr

        numerator = np.trapz(y=bandpass * np.array(I),
                                x=freq_array, dx=del_freq)

        sz_signal = numerator / (bandpass_int) # SZ model
            
        # Dust emission treatment    
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
        # dust_signal *= beta_density_map_2d

        total_model = beta_density_map_2d * (sz_signal + dust_signal)
        
        return total_model

    def initialize(self, theta):

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


