import numpy as np

def rebin(xs, ys, cell_size=16):  # Returns data rebinned to given cell size
    
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

def filtr(ys, cell_size): # Returns data with filtered out peak (center) points for rb_size binned

    for i in range(cell_size//2-1, len(ys), cell_size):
        ys[i] = np.mean([ys[i-1], (ys[i+2])])
        ys[i+1] = np.mean([ys[i], (ys[i+2])])
        ys[i+2] = np.mean([ys[i+1], (ys[i+3])])

    return ys

def comb_spec(ys, comb):  # Returns data divided by comb

    ys = np.array(ys)
    return ys / np.resize(comb, ys.shape)

def clean(xs, ys, cell_size=16, low_freq=1024.01, high_freq=1923.25):  # Returns selected region from low_freq to high_freq

    xs = np.array(xs)
    i1 = np.argmin(np.abs(xs - low_freq)) // cell_size * cell_size
    i2 = np.argmin(np.abs(xs - high_freq)) // cell_size * cell_size
    i1, i2 = min(i1, i2), max(i1, i2)

    xs_cut = xs[i1:i2].copy()
    ys_cut = ys[i1:i2].copy()

    return xs_cut, ys_cut