import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as consts
import scipy.interpolate as interp

class SWH_updater:
    def __init__(self, cal_file):
        self.cal_ds = xr.open_dataset(cal_file)
        self.subcycle_length = 146
        
    def get_csubc_from_cp(self, c,p):
        return 'cycle_{}_subcycle_{}'.format(c,(p-1)//self.subcycle_length+1)
    
    def correct_SWH(self, input_data_2km, Re=6371.0e3, baseline=10.048, lambda_c=consts.c/35.75e9):
        crid = input_data_2km.crid
        if crid not in ['PIC2', 'PID0']:
            raise ValueError('Only versions PIC2 or PID0 of the products can be calibrated in post-processing in this way')
        
        yaw_flip = np.all(np.abs(input_data_2km.sc_yaw.values)>5.)
        sign_flip_xtd = {True:-1, False:1}[yaw_flip]
        try:
            cal_values = self.cal_ds[self.get_csubc_from_cp(input_data_2km.cycle_number, input_data_2km.pass_number)].values
        except:
            raise ValueError('No calibration applicable to this cycle/pass is available in the calibration file')
        
        gamma_cal_corr_I = interp.interp1d(sign_flip_xtd*self.cal_ds.xtrack_bin.values, cal_values, bounds_error=False, fill_value='extrapolate')


        H_sat = input_data_2km.sc_altitude.values
        curv_fact = H_sat/(Re+H_sat)

        alpha_0 = np.abs(input_data_2km.cross_track_distance.values.T/Re)
        r0 = np.sqrt( (Re+H_sat)**2 + Re**2 - 2*Re*(Re+H_sat)*np.cos(alpha_0) )

        kappa_0 = (baseline/lambda_c/r0/np.tan(alpha_0)*curv_fact).T
        gamma_vol_corrected = input_data_2km.volumetric_correlation.values/(1+gamma_cal_corr_I(input_data_2km.cross_track_distance.values))
        SWH = 2/np.pi/kappa_0*np.sqrt(-2*np.log(np.clip(gamma_vol_corrected, 0,1)))

        input_data_2km = input_data_2km.assign(swh_karin_corrected=(["num_lines", "num_pixels"],SWH))
        return input_data_2km