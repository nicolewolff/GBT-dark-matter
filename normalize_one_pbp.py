import sys, os
sys.path.insert(0, '..') 
import numpy as np
from scipy.optimize import curve_fit

v_virial = 250
c = 2.9979e5

def normalize(xs, ys, window):
    freq_diff = xs[1] - xs[0]
    ind_xs = np.arange(-(window//2), window//2+1)
    lower = (window - 1) // 2
    upper = (window + 1) // 2
    #sigma = .79
    C = 2.9979e5
    ys_mean = ys / ys.mean()
    chi_squareds = []
    for i in range(0, len(ys)-16, 16):
        ys_mean[i] = np.NaN
        ys_mean[i+15] = np.NaN
        ys_mean[i+1] = np.NaN
        ys_mean[i+14] = np.NaN
    
    
    def fit_func(x, A, b,c, d, f, g, center):
        y = (A**2)*np.exp((-(freq_diff*x-center)**2)/(2*sigma**2)) + b + c*x + d*x**2 + f*x**3 + g*x**4
        #print('sigma:' + str(sigma))
        return y
    normalized_spectrum = []

    for i in range(lower, len(xs)-upper):

        sigma = v_virial*xs[i]/(C*np.sqrt(6))
      
        current_ys = ys_mean[i-lower:i+upper]
        idx = np.isfinite(current_ys)
        
    #fitting procedure
        parameters, covariance = curve_fit(fit_func, ind_xs[idx], current_ys[idx], [2, 1,0,0,0,0, 0], maxfev=10000)

        fit_A, fit_b, fit_c, fit_d, fit_f, fit_g, fit_center = parameters
        fit_ys = fit_func(ind_xs, fit_A, fit_b,fit_c, fit_d, fit_f,  fit_g, fit_center)
        chi_squared = np.sum((current_ys[idx] - fit_ys[idx])**2/fit_ys[idx])

        chi_squareds.append(chi_squared)
       
        #gaussian + polynomial
        #g_plus_p = fit_func(0, fit_A, fit_b,fit_c, fit_d, fit_f,  fit_g, fit_center)
        g = fit_func(0, fit_A, 0, 0, 0, 0, 0,fit_center)
        p = fit_func(0, 0, fit_b, fit_c, fit_d, fit_f,fit_g,fit_center)  
        #normalized_spectrum.append(g_plus_p/p)
        if np.isfinite(chi_squared) and chi_squared > 0:
            normalized_spectrum.append((1/(1+np.exp((np.log10(chi_squared) + 4)/.5)))*g/p + 1)

        else:
            normalized_spectrum.append(1)
    # new_xs = xs[lower: len(xs)-upper]
    # np.save('/home/dataadmin/GBTData/SharedDataDirectory/sband/normalized_spectra_pbp/normed_freqs_pbp.npy', new_xs)
    print("normed")
    np.save('/home/dataadmin/GBTData/SharedDataDirectory/lband/chi_squareds.npy', chi_squareds)
    return normalized_spectrum
  
def main(worker_id, filepath, out_filepath,window): #change path of output logs
    if not os.path.exists("NormOutputLogs"): 
        os.makedirs("NormOutputLogs")
    
    sys.stdout = open(os.path.join('NormOutputLogs', f"{worker_id}.out"), "w")
    sys.stderr = open(os.path.join('NormOutputLogs', f"{worker_id}.err"), "w")
    
    import time 
    try: 
        start = time.time() 
        print(f"loading files...")
        freqs, spec = np.load(filepath, allow_pickle=True)
        end = time.time()
        print(f"Loaded files in  {end - start}\nTime To normalize...")
        normed = normalize(freqs, spec, window)
        print(f"Normalized in {time.time() - end}")
        np.save(out_filepath, normed)
        print(f"Saved and Finished! Total Time {time.time() - start}")
        os.chmod(out_filepath, 0o666)
    except Exception as e: 
        print(e,  file=sys.stderr)
