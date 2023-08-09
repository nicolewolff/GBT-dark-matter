from signal import SIG_BLOCK
import sys, os
sys.path.insert(0, '..')
import pandas as pd
import numpy as np
from tqdm import tqdm

inject_path = '/home/dataadmin/GBTData/SharedDataDirectory/lband/raw_files_npy'
data_table_path = '/home/dataadmin/GBTData/SharedDataDirectory/lband/data_table/all_info.csv'

is_decay = False
template = False

v_virial = 250
v_earth = 225
sig_loc = 1390
c = 299792
decay_rate = 0
dm_profile = 'nfw'


"""
:param sig_loc: Center frequency ued to calculate signal magnitude
:type sig_loc: float
:param sig_std: Doppler broadening of the source
:type sig_std: float
:param sig_amp: Weight given to a signal's amplitude depending on DM density profile and annihilation/decay formula
:type sig_amp: float
:return: power spectrum with an added Gaussian signal every 50 MHz 
:rtype: float
"""
def inject(xs, ys, theta, sig_loc, sig_std, sig_amp):
    shift_factor = get_shift_factor(theta)
    sig_loc, sig_std, sig_amp = sig_loc*shift_factor, sig_std*shift_factor, sig_amp/shift_factor
    return (ys + sig_amp*np.exp(-(xs - sig_loc)**2 / (2*sig_std**2)))  # Gaussian signal
    
def get_shift_factor(theta):
    
    beta = v_earth / c
    radians_from_v_earth = np.radians(theta)
    return 1 + beta*np.cos(radians_from_v_earth)

def inject_spaced(ann_cross_sec, loc, filepath_to_save, template):
    try:
        os.makedirs(filepath_to_save, exist_ok=False)
        if is_decay:
            sig_std = v_virial*sig_loc/(c*np.sqrt(3))
            sig_amp_ratio = 2.75e23*decay_rate / (v_virial*(sig_loc/1e3)**4)
            col_name = 'max_normed_'+dm_profile
        else:
            sig_std = v_virial*sig_loc/(c*np.sqrt(6))
            sig_amp_ratio = 3.55e28*ann_cross_sec / (v_virial*(sig_loc/1e3)**4) # convert MHz to GHz
            col_name ='nfw_sq'
        targ_info = pd.read_csv('/home/dataadmin/GBTData/SharedDataDirectory/lband/data_table/all_info.csv').set_index('file_name')
        names = os.listdir(inject_path)
        for name in tqdm(names):
            try:
                spec =  np.load(os.path.join(inject_path, name))[1]
                if np.shape(spec)[0] == 315392:
                    freq = np.load(os.path.join(inject_path, name))[0]
                    if template:
                        spec = np.ones_like(spec)                  
                    spec_med = np.median(spec)
                    theta = targ_info.loc[name[:-4], 'sep90']
                    amp_ratio = sig_amp_ratio*targ_info.loc[name[:-4], col_name]
                    
                    amp = spec_med*amp_ratio
                    loc_loop = loc
                    end = freq[freq.size-50]
                    salted = inject(freq, spec, theta, loc_loop, sig_std, amp)
                    while loc_loop <= end-50:  # loop every 50 MHz
                        loc_loop = loc_loop + 50
                        salted = inject(freq, salted, theta, loc_loop, sig_std, amp)
                
                    save_loc_spec = os.path.join(filepath_to_save, name)
                    save_loc_freq = os.path.join(filepath_to_save, name[:-4] + "_freqs.npy")
                    np.save(save_loc_spec, salted)
                    np.save(save_loc_freq, freq)
                
            except Exception as e:
                print(e)
    except Exception as e:
        print(e)
