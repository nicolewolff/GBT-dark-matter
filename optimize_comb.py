import numpy as np
from numpy import polyval
import matplotlib.pyplot as plt
import os
import pandas as pd
import math
from scipy import interpolate, signal
from tqdm import tqdm
from astropy.stats.sigma_clipping import sigma_clip
import itertools, json
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize, curve_fit

v_virial = 250
c = 2.9979e5
comb_path = '/home/dataadmin/GBTData/SharedDataDirectory/lband/data_info/optimized_unflattened_comb.npy'

""" Polynomial fit to detrend spectrum with the Scipy library curve_fit function

:param xs: frequencies
:type xs: ndarray
:param ys: powers
:type ys: ndarray
:return: normalized powers 
:rtype: ndarray

"""

def normalize(xs, ys, order, window):
    xs, ys = np.array(xs), np.array(ys)
    ys = ys / ys.mean()
    ys_mean = ys / ys.mean()
    for i in range(0, len(ys)-16, 16):  # Excise unstable points around minima in a coarse channel 
        ys_mean[i] = np.NaN
        xs[i] = np.NaN
        ys_mean[i+15] = np.NaN
        xs[i+15] = np.NaN
        ys_mean[i+1] = np.NaN
        xs[i+1] = np.NaN
        ys_mean[i+14] = np.NaN
        xs[i+14] = np.NaN
    ys_mean = ys_mean[~np.isnan(ys_mean)]
    xs = xs[~np.isnan(xs)]
    normalized_spectrum = []

    def poly_norm(xs, ys, params):
        order, window = params
        ind_xs = np.arange(-(window//2), window//2+1)
        lower = (window - 1) // 2
        upper = (window + 1) // 2
        
        
        fitted_ys = np.zeros(ys.shape)
        for i in range(0,len(xs)):
            if i <= lower: 
                current_ys = ys_mean[i:i+window]
                idx = np.isfinite(current_ys)
                c_w = np.polyfit(ind_xs[idx], current_ys[idx], order)
                fitted_ys[i] = c_w[-1]
                normalized_spectrum.append(ys[i] / np.polyval(c_w,ind_xs)[0])

            elif i >= len(xs) - upper:
                current_ys = ys_mean[i-window+1:i+1]
                idx = np.isfinite(current_ys)
                c_w = np.polyfit(ind_xs[idx], current_ys[idx], order)
                fitted_ys[i] = c_w[-1]
                normalized_spectrum.append(ys[i] / np.polyval(c_w,ind_xs)[-1])

            else:
                current_ys = ys_mean[i-lower:i+upper]
                idx = np.isfinite(current_ys)
                c_w = np.polyfit(ind_xs[idx], current_ys[idx], order)
                fitted_ys[i] = c_w[-1]
                normalized_spectrum.append(ys[i] / np.polyval(c_w,ind_xs)[lower])

        return xs, fitted_ys
    xs, fitted_ys = poly_norm(xs,ys,[order,window])
    return xs, fitted_ys 

def filtr(ys, cell_size):
    """
    Returns data with filtered out peak (center) points for rb_size binned
    """
    for i in range(cell_size//2-1, len(ys), cell_size):
        ys[i] = np.mean([ys[i-1], (ys[i+2])])
        ys[i+1] = np.mean([ys[i], (ys[i+2])])
        ys[i+2] = np.mean([ys[i+1], (ys[i+3])])

    return ys

def create_comb(xs_injected,filtered_normed):
    ## Define the 1750-1800 MHz region used to create comb ##
    print(len(filtered_normed))
    optimized_combs = []

    #for i in range(11):
    start = 16*248
    end = start + 18*16
    quiet_section_powers = filtered_normed[start:end]
    init_comb = np.array(quiet_section_powers[0:16])
    print("init comb: ",init_comb)

    ## Optimize a 16-point comb by minimizing the standard deviation ##

    def calculate_stdev(comb, rebin_data):
        print(len(rebin_data))
        channels = np.split(rebin_data, len(rebin_data))
        stdevs = []
        for channel in channels:
            combed = channel / comb
            stdevs.append(np.std(combed))
        return np.mean(stdevs)

    def optimize_comb(data):
        def function(comb):
            return calculate_stdev(comb,data)
        
        result = minimize(function, init_comb, method = 'L-BFGS-B')
        best_comb = result.x
        
        return best_comb, result.fun

    data = np.reshape(quiet_section_powers, (18, 16))
    best_comb, min_stddev = optimize_comb(data)
    best_comb = best_comb / np.mean(best_comb[2:14])
    #optimized_combs.append(best_comb)
    #average_comb = np.mean(optimized_combs,axis=0)
    #print("average comb: ",average_comb)
    np.save(comb_path,best_comb)
    return best_comb

def rebin(xs, ys, cell_size=16):
    """
    Returns data rebinned to given cell size
    """
    rebin_size=1024//cell_size
    rebinned_spec = []
    for i in range(len(ys)//rebin_size):
        average = np.mean(ys[i*rebin_size:i*rebin_size+rebin_size])
        rebinned_spec.append(average)

    rebinned_freq = []
    for i in range(len(xs)//rebin_size):
        average = np.mean(xs[i*rebin_size:i*rebin_size+rebin_size])
        rebinned_freq.append(average)
    return rebinned_freq, rebinned_spec

def comb_spec(ys, comb):
    """
    Returns combed data
    """
    ys = np.array(ys)

    return ys / np.resize(comb, ys.shape)

def clean(xs, ys, cell_size=16, low_freq=1024.01, high_freq=1923.25):
    """
    Returns selected region from low_freq to high_freq
    """
    xs = np.array(xs)
    i1 = np.argmin(np.abs(xs - low_freq)) // cell_size * cell_size
    i2 = np.argmin(np.abs(xs - high_freq)) // cell_size * cell_size
    i1, i2 = min(i1, i2), max(i1, i2)

    xs_cut = xs[i1:i2].copy()
    ys_cut = ys[i1:i2].copy()

    return xs_cut, ys_cut

def main(filepath, out_filepath):
    if 'freqs' not in filepath:
        try:
            ys_injected =  np.load(filepath)
            xs_injected = np.load(filepath[:-4]+ "_freqs.npy")
            filtered = filtr(ys_injected, 1024)
            xs, ys = rebin(xs_injected, filtered,16)
            xs, normed = normalize(xs, ys, 6, 241) # 241 because 16 points/channel * 45 MHz total / 3 MHz/channel
            # xs_1, ys_1 = clean(xs, ys, cell_size=16, low_freq=1024.01, high_freq=1229.1)
            # xs_2, ys_2 = clean(xs, ys, cell_size=16, low_freq=1302.4, high_freq=1525)
            # xs_3, ys_3 = clean(xs, ys_1, cell_size=16, low_freq=1560.2, high_freq=1923.25)
            # xs = np.concatenate((xs_1, xs_2, xs_3))
            # ys = np.concatenate((ys_1, ys_2, ys_3))
            # create new comb after cleaning messy regions (1229-1302, 1525-1560)
            best_comb = create_comb(xs,normed)
            print('best comb:',best_comb)
        
            if len(ys) == 4928:  # File contains all coarse channels
                ys_new = normed 
                xs_1, ys_1 = clean(xs, ys_new, cell_size=16, low_freq=1024.01, high_freq=1229.1)

                xs_2, ys_2 = clean(xs, ys_new, cell_size=16, low_freq=1302.4, high_freq=1525)

                xs_3, ys_3 = clean(xs, ys_new, cell_size=16, low_freq=1560.2, high_freq=1923.25)

                xs = np.concatenate((xs_1, xs_2, xs_3))

                ys = np.concatenate((ys_1, ys_2, ys_3))
                data = np.array([xs, ys])
                np.save(out_filepath, data)
        except Exception as e:
            print(e)

in_path = '/home/dataadmin/GBTData/SharedDataDirectory/lband/injected_nicole/start_1037/signal_0.0/GBT_57592_53807_HIP18267_mid.npy'
out_path = '/home/nicolew/preprocessed_best_comb.npy'

if __name__ == "__main__":
    main(in_path,out_path)
