import sys, os
sys.path.insert(0, '..')
import pandas as pd
import numpy as np

comb_path = '/home/dataadmin/GBTData/SharedDataDirectory/lband/data_info/optimized_unflattened_comb.npy'

def rebin(xs, ys, cell_size=16):
    """ Rebins data to a smaller number of points per coarse channel

    :param xs: frequencies
    :type xs: 1d array
    :param ys: powers
    :type ys: 1darray
    :param cell_size: number of desired points in a coarse channel (16 for this analysis)
    :type cell_size: int
    :return: rebinned frequency array, rebinned power array
    :rtype: tuple
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

def filter(ys, cell_size):
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
    comb = np.load(comb_path, allow_pickle=True)
    if 'freqs' not in filepath:     
        try:
            spec =  np.load(filepath)
            freq = np.load(filepath[:-4]+ "_freqs.npy")
            ys = filter(spec, 1024)
            xs, ys = rebin(freq, ys, 16)
            if len(ys) == 4928: 
                if not template:
                    ys = comb_spec(ys, comb)
                xs_1, ys_1 = clean(xs, ys, 16, 1024.01, 1229.1)                
                xs_2, ys_2 = clean(xs, ys, 16, 1302.4, 1525)
                xs_3, ys_3 = clean(xs, ys, 16, 1560.2, 1923.25)
                xs = np.concatenate((xs_1, xs_2, xs_3))
                ys = np.concatenate((ys_1, ys_2, ys_3))
                data = np.array([xs, ys])
                np.save(out_filepath, data)            
        except Exception as e:
            print(e)
                
