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
comb_path = '/home/dataadmin/GBTData/SharedDataDirectory/lband/data_info/polynomial_normalized_comb_new.npy'

def filtr(ys, cell_size):
    """ Removes center DC spike
    
    :param ys: powers
    :type ys: 1darray
    :param cell_size: number of points in a coarse channel based on the data type (medium resolution, high spectral resolution, high time resolution)
    :type cell_size: int
    :param window: number of points to fit polynomial to
    :type window: int
    :return: filtered powers
    :rtype: 1darray
    """
    for i in range(cell_size//2-1, len(ys), cell_size):
        ys[i] = np.mean([ys[i-1], (ys[i+2])])
    return ys

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

def normalize(xs, ys, order, window):
    """ Normalizes spectra by fitting and dividing by an unweighted polynomial
    
    :param xs: frequencies
    :type xs: 1darray
    :param ys: powers
    :type ys: 1darray
    :param order: order of polynomial
    :type order: int
    :param window: number of points to fit polynomial to
    :type window: int
    :return: normalized powers
    :rtype: 1darray
    """
    xs, ys = np.array(xs), np.array(ys)
    ys = ys / ys.mean()
    ys_mean = ys / ys.mean()
    for i in range(0, len(ys)-16, 16):  # Excise the 4 unstable valley points in each coarse channel 
        ys_mean[i] = np.NaN
        ys_mean[i+15] = np.NaN
        ys_mean[i+1] = np.NaN
        ys_mean[i+14] = np.NaN

    normalized_spectrum = []

    ind_xs = np.arange(-(window//2), window//2+1)
    lower = (window - 1) // 2
    upper = (window + 1) // 2  
    for i in range(0,len(xs)):
        if i <= lower:  # For points in the first 50 MHz, fit a polynomial to the right of the point 
            current_ys = ys_mean[i:i+window]
            idx = np.isfinite(current_ys)
            c_w = np.polyfit(ind_xs[idx], current_ys[idx], order)
            normalized_spectrum.append(ys[i] / np.polyval(c_w,ind_xs)[0])

        elif i >= len(xs) - upper:  # For points in the last 50 MHz, fit a polynomial to the left of the point
            current_ys = ys_mean[i-window+1:i+1]
            idx = np.isfinite(current_ys)
            c_w = np.polyfit(ind_xs[idx], current_ys[idx], order)
            normalized_spectrum.append(ys[i] / np.polyval(c_w,ind_xs)[-1])

        else:
            current_ys = ys_mean[i-lower:i+upper]  # For points in the middle, fit a polynomial to 25 MHz on either side
            idx = np.isfinite(current_ys)
            c_w = np.polyfit(ind_xs[idx], current_ys[idx], order)
            normalized_spectrum.append(ys[i] / np.polyval(c_w,ind_xs)[lower])
    return normalized_spectrum

def comb_spec(ys, comb):
    """ Removes the bandpass structure from data
    
    :param ys: powers
    :type ys: 1darray
    :param comb: 16-point comb
    :type ys: 1darray
    :return: combed powers
    :rtype: 1darray
    """
    ys = np.array(ys)
    return ys / np.resize(comb, ys.shape)

def clean(xs, ys, cell_size, low_freq, high_freq):
    """ Removes noisiest regions of the band corresponding to the notch filter
    
    :param xs: frequencies
    :type xs: 1darray
    :param ys: powers
    :type ys: 1darray
    :param cell_size: number of points per coarse channel after rebinning
    :type cell_size: int
    :param low_freq: the lowest frequency to keep for the desired region
    :type low_freq: float
    :param high_freq: the highest frequency to keep for the desired region
    :type high_freq: float
    :return: array of frequencies to keep, array of powers to keep
    :rtype: tuple
    """
    xs = np.array(xs)
    i1 = np.argmin(np.abs(xs - low_freq)) // cell_size * cell_size
    i2 = np.argmin(np.abs(xs - high_freq)) // cell_size * cell_size
    i1, i2 = min(i1, i2), max(i1, i2)
    xs_cut = xs[i1:i2].copy()
    ys_cut = ys[i1:i2].copy()
    return xs_cut, ys_cut

def main(worker_id, filepath, out_filepath, template):
    if 'freqs' not in filepath:  # The file is a power array
        try:
            ys_injected =  np.load(filepath)
            xs_injected = np.load(filepath[:-4]+ "_freqs.npy")
            filtered = filtr(ys_injected, 1024)
            xs, ys = rebin(xs_injected, filtered,16)
            # To calculate window: # points/coarse channel * frequency range of window [MHz] * 2.9 MHz/channel
            # Example: 16 [points/channel] * 53 [MHz] / 2.9 [MHz/channel] = 293 points
            # Note: window size must be odd in order to have an exact center
            normed = normalize(xs, ys, 5, 293)
            best_comb = np.load(comb_path)        
            if len(ys) == 4928:  # File contains all 308 coarse channels 
                ys_new = comb_spec(normed,best_comb)
                xs_1, ys_1 = clean(xs, ys_new, 16, 1024.01, 1229.1)
                xs_2, ys_2 = clean(xs, ys_new, 16, 1302.4, 1525)
                xs_3, ys_3 = clean(xs, ys_new, 16, 1560.2, 1923.25)
                xs, ys = np.concatenate((xs_1, xs_2, xs_3)), np.concatenate((ys_1, ys_2, ys_3))
                data = np.array([xs, ys])
                np.save(out_filepath, data)
        except Exception as e:
            print(e)
