## Author: Aya Keller ##

from signal import SIG_BLOCK
import sys, os
sys.path.insert(0, '..')
import pandas as pd
import numpy as np

filepath_to_inject = '/home/dataadmin/GBTData/SharedDataDirectory/sband/averaged_spectra'
filepath_to_save = '/home/dataadmin/GBTData/SharedDataDirectory/sband/signal_0'
os.makedirs(filepath_to_save, exist_ok = True)
is_decay = False

v_virial = 250
v_earth = 225
sig_loc = 2235
ann_cross_sec = 0
decay_rate = 0
dm_profile = 'nwf'
template = False

def inject(xs, ys, theta, v_earth, sig_loc, sig_std, sig_amp):
    shift_factor = get_shift_factor(theta, v_earth)
    sig_loc, sig_std = sig_loc*shift_factor, sig_std*shift_factor
    sig_amp = sig_amp / shift_factor
    return (ys + sig_amp*np.exp(-(xs - sig_loc)**2 / (2*sig_std**2)))

def get_shift_factor(theta, v_earth):
    c = 299792
    beta = v_earth / c
    radians_from_v_earth = np.radians(theta)
    return 1 + beta*np.cos(radians_from_v_earth)

def main():

    C = 2.9979e5
    if is_decay:
        sig_std = v_virial*sig_loc/(C*np.sqrt(3))
        sig_amp_ratio = 2.75e23*decay_rate / (v_virial*(sig_loc/1e3)**4) # convert MHz to GHz
        col_name = dm_profile+'_integ_decay'
    else:
        sig_std = v_virial*sig_loc/(C*np.sqrt(6))
        sig_amp_ratio = 7.79e28*ann_cross_sec / (v_virial*(sig_loc/1e3)**4) # convert MHz to GHz
        col_name = dm_profile+'_integ_anhil'
    targ_info = pd.read_csv('/home/dataadmin/GBTData/SharedDataDirectory/sband/run_data_sband/data_table.csv').set_index('filename')
    names = os.listdir(filepath_to_inject)
    # import pdb; 
    for name in names:
        # pdb.set_trace()
        try:
            if "freq" not in name:
                spec =  np.load(os.path.join(filepath_to_inject, name))
                freq = np.load(os.path.join(filepath_to_inject, name[:-4] + "_freqs.npy"))[::-1]
                if template:
                    spec = np.ones_like(spec)
                spec_med = np.median(spec)
                theta = targ_info.loc[name[:-4]+'.h5', 'theta']
                # print(targ_info.loc[name[:-4]+'.h5', col_name])
                amp_ratio = sig_amp_ratio*targ_info.loc[name[:-4]+'.h5', col_name]

                amp = spec_med*amp_ratio

                salted = inject(freq, spec, theta, v_earth, sig_loc, sig_std, amp)

                save_loc_spec = os.path.join(filepath_to_save, name)
                save_loc_freq = os.path.join(filepath_to_save, name[:-4] + "_freqs.npy")
                np.save(save_loc_spec, salted)
                np.save(save_loc_freq, freq)
        except Exception as e:
          print(e)

main()
