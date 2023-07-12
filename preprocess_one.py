import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import math
from scipy import interpolate
from scipy import signal
from scipy.interpolate import UnivariateSpline
from tqdm import tqdm
from astropy.stats.sigma_clipping import sigma_clip
import itertools, json
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

v_virial = 250
c = 2.9979e5
comb = np.load('/home/dataadmin/GBTData/SharedDataDirectory/lband/data_info/comb_nicole.npy')

## Spline to detrend the spectrum ##

def findline(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    m = (y2 - y1)/(x2 - x1) 
    b = y1 - m*x1 
    return m, b

def pad_data(x, y, n_pad = int(5)):
    first_two_x = x[0:2]
    first_two_y = y[0:2]
    first_two_points = np.dstack([first_two_x, first_two_y])[0]
    left_m, left_b = findline(first_two_points[0], first_two_points[1])

    last_two_x = x[-2:]
    last_two_y = y[-2:]
    last_two_points = np.dstack([last_two_x, last_two_y])[0]
    right_m, right_b = findline(last_two_points[0], last_two_points[1])

    left_pad_x = np.array(x[0:n_pad]) - (x[n_pad]-x[0])
    left_pad_y = left_m*left_pad_x + left_b
    right_pad_x = np.array(x[-n_pad:]) + (x[n_pad]-x[0])
    right_pad_y = right_m*right_pad_x + right_b

    padded_x = np.array(list(itertools.chain(left_pad_x, x, right_pad_x)))
    padded_y = np.array(list(itertools.chain(left_pad_y, y, right_pad_y)))

    return padded_x, padded_y

def CC_points(x, y):
    x = np.array(x)
    y = np.array(y)
    num_CCs = int(len(x)/1024) #The number of coarse channels, since each one has length 1024
    
    coarsechans = np.split(np.arange(len(x)), num_CCs)

    yvals = []
    xvals = []
    for chan_idx in coarsechans:
        chanfreqs = x[chan_idx] #Frequencies for this coarse channel
        chanpowers = y[chan_idx] #Power values for this coarse channel
        
        power1 = np.percentile(chanpowers[0:255], 20)
        power2 = np.percentile(chanpowers, 10)
        power3 = np.percentile(chanpowers[-255:], 20)
        
        powers = [power1, power2, power3]
        freqs = np.percentile(chanfreqs, [10,50,90])
        
        for f in freqs:
            xvals.append(f)
        
        for p in powers:
            yvals.append(p)
                                              
    xvals, yvals = pad_data(xvals, yvals)
    return xvals, yvals

def clean_CC_points(xvals, yvals):                                                                                           
    for i in np.arange(5):                                                                                         
        for i in np.arange(1, len(xvals)-1):
            previousp = np.array([xvals[i-1], yvals[i-1]]) #Point on the left
            currentp = np.array([xvals[i], yvals[i]]) #Current points in question
            nextp = np.array([xvals[i+1], yvals[i+1]]) #point on the right
            m, b = findline(previousp, nextp) #slope and intercept of a line connecting left and right points
            interpolatedp = np.array([xvals[i], m*xvals[i]+b]) #what the p
            if (currentp[1] > previousp[1] and currentp[1] > nextp[1]) and \
            currentp[1]-interpolatedp[1] > .01*np.linalg.norm(currentp - previousp) \
            and currentp[1] > interpolatedp[1]:                                                  
                m, b = findline(previousp, nextp)
                yvals[i] = m*xvals[i]+b
    return xvals, yvals        

def trend_spline(x,y,k=3):
    initial_x, initial_y = CC_points(x, y)
    cleaned_x, cleaned_y = clean_CC_points(initial_x, initial_y)
                                                                                           
    N=len(x); rmserror=.00001
    s = N*(rmserror * np.fabs(cleaned_y).max())**2
    spl = UnivariateSpline(cleaned_x, cleaned_y, s=s, k=k)
    return cleaned_x, cleaned_y, spl

def filtr(ys, cell_size):
    """
    Returns data with filtered out peak (center) points for rb_size binned
    """
    for i in range(cell_size//2-1, len(ys), cell_size):
        ys[i] = np.mean([ys[i-1], (ys[i+2])])
        ys[i+1] = np.mean([ys[i], (ys[i+2])])
        ys[i+2] = np.mean([ys[i+1], (ys[i+3])])

    return ys

def create_comb(xs_injected,detrended_filtered):
    ## Define the 1750-1800 MHz region used to create comb ##

    start = -1 + 512*491 + 512 
    end = start + 36*512
    quiet_section_freqs = xs_injected[start:end]
    quiet_section_powers = detrended_filtered[start:end]
    quiet_section_rebin = np.array(rebin(quiet_section_freqs,quiet_section_powers))
    init_comb = np.array(quiet_section_rebin[1][0:16])

    ## Optimize a 16-point comb by minimizing the standard deviation ##

    def calculate_stdev(comb, rebin_data):
        channels = np.split(rebin_data, len(rebin_data)//16)
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

    data = np.reshape(quiet_section_rebin[1], (18, 16))
    best_comb, min_std_dev = optimize_comb(data)

    return best_comb, min_std_dev

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

def main(worker_id, filepath, out_filepath, template):
    if 'freqs' not in filepath:
        try:
            ys_injected =  np.load(filepath)
            xs_injected = np.load(filepath[:-4]+ "_freqs.npy")
            normed = normalize(xs_injected,ys_injected,.1,1,5,55)

            xvals, yvals = CC_points(xs_injected, np.log10(ys_injected))
            xvals, yvals = clean_CC_points(xvals, yvals)

            x, y, spline = trend_spline(xs_injected, np.log10(ys_injected))

            detrended = np.log10(ys_injected) / spline(xs_injected)

            detrended_filtered = filtr(detrended, 1024)

            xs, ys = rebin(x, detrended_filtered,16)
            
            if len(ys) == 4928: 

                ys_new = comb_spec(ys,comb)

                xs_1, ys_1 = clean(xs, ys_new, cell_size=16, low_freq=1024.01, high_freq=1229.1)

                xs_2, ys_2 = clean(xs, ys_new, cell_size=16, low_freq=1302.4, high_freq=1525)

                xs_3, ys_3 = clean(xs, ys_new, cell_size=16, low_freq=1560.2, high_freq=1923.25)

                xs = np.concatenate((xs_1, xs_2, xs_3))

                ys = np.concatenate((ys_1, ys_2, ys_3))

                data = np.array([xs, ys])
                np.save(out_filepath, data)
        except Exception as e:
            print(e)
