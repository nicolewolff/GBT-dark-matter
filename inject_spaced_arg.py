from signal import SIG_BLOCK
import sys, os
sys.path.insert(0, '..')
import pandas as pd
import numpy as np
#from joblib import Parallel, delayed
#run with big signal inkect ~1e42 and then plot on top of another spectra 
#

filepath_to_inject = '/home/dataadmin/GBTData/SharedDataDirectory/lband/raw_files_npy'

is_decay = False

v_virial = 250
v_earth = 225
sig_loc = 1390

decay_rate = 0
dm_profile = 'nfw'
#template = False #run it with template: MAKET THIS AN INPUT VARIABLE

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

def inject_spaced(ann_cross_sec, loc, filepath_to_save, template):
    os.makedirs(filepath_to_save, exist_ok=True)
    C = 2.9979e5
    if is_decay:
        sig_std = v_virial*sig_loc/(C*np.sqrt(3))
        sig_amp_ratio = 2.75e23*decay_rate / (v_virial*(sig_loc/1e3)**4) # convert MHz to GHz
        col_name = 'max_normed_'+dm_profile
    else:
        sig_std = v_virial*sig_loc/(C*np.sqrt(6))
        #change 7.79e28 to an L band value
        sig_amp_ratio = 3.55e28*ann_cross_sec / (v_virial*(sig_loc/1e3)**4) # convert MHz to GHz
        col_name ='nfw_sq'
    #switch targ_info to lband table and play around to get it in the same order,if error index out of range its a table mismatch error
    targ_info = pd.read_csv('/home/dataadmin/GBTData/SharedDataDirectory/lband/data_table/all_info.csv').set_index('file_name')
    #targ_info.set_index("file_name", inplace = True) 
    names = os.listdir(filepath_to_inject) #CHANGE THIS
    # import pdb; 
    for name in names:
        # pdb.set_trace()
        #set x
        try:
            spec =  np.load(os.path.join(filepath_to_inject, name))[1]
            #if "freq" not in name:
            #check if proper size
            if np.shape(spec)[0] == 315392:
                #these might change, it is forward not backwards
                #spec =  np.load(os.path.join(filepath_to_inject, name))[0]
                freq = np.load(os.path.join(filepath_to_inject, name))[0]
                if template:
                    spec = np.ones_like(spec)
                    
                spec_med = np.median(spec)
                #theta is def called sep_90 in other table
                #theta = targ_info.loc[name[:-4]+'.h5', 'sep90']
                theta = targ_info.loc[name[:-4], 'sep90']
                # print(targ_info.loc[name[:-4]+'.h5', col_name])
                amp_ratio = sig_amp_ratio*targ_info.loc[name[:-4], col_name]
                
                amp = spec_med*amp_ratio
                
                #loop variables
                loc_loop = loc
                end = freq[freq.size-50]
                salted = inject(freq, spec, theta, v_earth, loc_loop, sig_std, amp)
                #loop incrememnting by 50 Mhz until end
                while loc_loop <= end-50 :
                    loc_loop = loc_loop + 50
                    salted = inject(freq, salted, theta, v_earth, loc_loop, sig_std, amp)
            
                save_loc_spec = os.path.join(filepath_to_save, name)
                save_loc_freq = os.path.join(filepath_to_save, name[:-4] + "_freqs.npy")
                np.save(save_loc_spec, salted)
                np.save(save_loc_freq, freq)
               
        except Exception as e:
            #print(e) 
            
            pass  


  
