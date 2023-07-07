
import sys, subprocess, os
import numpy as np
import pandas as pd
import math
from scipy.interpolate import interp1d
from normalize_pbp_arg import *
from preprocess_arg import *
from inject_spaced_arg import *
from dateutil.relativedelta import relativedelta
import datetime
from logger import * 
import matplotlib.pyplot as plt
#comment out everything in the mega script except the inject part
def main_function(start_frequency, injected_signal):
    logger = get_logger()
    # variables to change
    # start_frequency = 1800
    # injected_signal= 1e-43 
    #injected_signal = injected_signal
    logger.debug(f"Starting process for {start_frequency} and {injected_signal}")
    start_time = datetime.datetime.now()
    #change this to have 0 signal 
    injected_filepath = f'/home/dataadmin/GBTData/SharedDataDirectory/lband/injected_limits_analysis/start_{start_frequency}/signal_{injected_signal}'
    preprocessed_filepath = f'/home/dataadmin/GBTData/SharedDataDirectory/lband/preprocessed_limits_analysis_uncombed/start_{start_frequency}/signal_{injected_signal}' 

    normalized_filepath = f'/home/dataadmin/GBTData/SharedDataDirectory/lband/normalized_spectra_spaced_uncombed/limit_analysis/start_{start_frequency}/signal_{injected_signal}' 
    normalized_uninjected_filepath = '/home/dataadmin/GBTData/SharedDataDirectory/lband/normalized_spectra_spaced_uncombed/limit_analysis/start_1037/signal_0' 
    template_filepath = '/home/dataadmin/GBTData/SharedDataDirectory/lband/limit_analysis_template_normalized'
    data_table = pd.read_csv('/home/dataadmin/GBTData/SharedDataDirectory/lband/data_table/all_info.csv').set_index('file_name')
    xs = np.load('/home/dataadmin/GBTData/SharedDataDirectory/lband/normalized_spectra_pbp/normed_freqs_pbp.npy')
    filepath_to_save = f'/home/dataadmin/GBTData/SharedDataDirectory/lband/limit_analysis_uncombed/start_{start_frequency}/signal_{injected_signal}' 

    # xs = np.load('/home/dataadmin/GBTData/SharedDataDirectory/sband/normalized_spectra_quartic/normed_freqs_quartic.npy')
    # normalized_filepath = f'/home/dataadmin/GBTData/SharedDataDirectory/lband/normalized_spectra_quartic_interpolated/signal_{injected_signal}' 
    # normalized_uninjected_filepath = '/home/dataadmin/GBTData/SharedDataDirectory/sband/normalized_spectra_quartic_interpolated/signal_0' 
    # template for preprocess and normalized
    normalized_template_filepath = '/home/dataadmin/GBTData/SharedDataDirectory/lband/normalized_spectra_spaced/template' #template for normalized 
    injected_template_filepath = f'/home/dataadmin/GBTData/SharedDataDirectory/lband/injected_limits_analysis/template'
    preprocessed_template_filepath = f'/home/dataadmin/GBTData/SharedDataDirectory/lband/preprocessed_limits_analysis/template' 

    # filepath_to_save = f'/home/dataadmin/GBTData/SharedDataDirectory/sband/quartic_interpolated_results/signal_{injected_signal}' 


    #UNCOMMENT
    # Injection (comment all of these out but this one)
    # inject_start_time = datetime.datetime.now()
    # inject_spaced(injected_signal, start_frequency, injected_filepath, False)
    # inject_end_time = datetime.datetime.now()
    # delta_inject = relativedelta(inject_end_time, inject_start_time)
    # logger.debug(f'Injection done in {delta_inject.hours} hrs, {delta_inject.minutes} mins, {delta_inject.seconds} sec')
    # print("done inject")
    #UNCOMMENT
    # Preprocessing
    # preprocess_start_time = datetime.datetime.now()
    # preprocess(injected_filepath, preprocessed_filepath, False)
    # preprocess_end_time = datetime.datetime.now()
    # delta_preprocess = relativedelta(preprocess_end_time, preprocess_start_time)
    # logger.debug(f'Preprocessing done in {delta_preprocess.hours} hrs, {delta_preprocess.minutes} mins, {delta_preprocess.seconds} sec')
    # print("done preprocess")

    # # Uninjected normalization
    # norm_start_time = datetime.datetime.now()
    # normalize('/home/dataadmin/GBTData/SharedDataDirectory/lband/signal_0_preprocessed_uncombed', normalized_uninjected_filepath)
    # norm_end_time = datetime.datetime.now()
    # delta_norm = relativedelta(norm_end_time, norm_start_time)
    # logger.debug(f'Uninjected normalization done in {delta_norm.hours} hrs, {delta_norm.minutes} mins, {delta_norm.seconds} sec')
    # print("done uninject norm")
    #UNCOMMENT
    # Normalization
    # norm_start_time = datetime.datetime.now()
    # print('starting norm')
    # normalize(preprocessed_filepath, normalized_filepath)
    # norm_end_time = datetime.datetime.now()
    # delta_norm = relativedelta(norm_end_time, norm_start_time)
    # logger.debug(f'Normalization done in {delta_norm.hours} hrs, {delta_norm.minutes} mins, {delta_norm.seconds} sec')
    # print("done norm")
    #UNCOMMENT
    # Injection Template (comment all of these out but this one) MAKE TEMPLATE VARIABLE A PARAMETER, create new paths and fix
    #inject_start_time = datetime.datetime.now()
    #inject_spaced(injected_signal, start_frequency, injected_template_filepath, True)
    #inject_end_time = datetime.datetime.now()
    #delta_inject = relativedelta(inject_end_time, inject_start_time)
    #logger.debug(f'Injection Template done in {delta_inject.hours} hrs, {delta_inject.minutes} mins, {delta_inject.seconds} sec')
#UNCOMMENT
    # Preprocessing Template
    #preprocess_start_time = datetime.datetime.now()
    #preprocess(injected_template_filepath, preprocessed_template_filepath, True)
    #preprocess_end_time = datetime.datetime.now()
    #delta_preprocess = relativedelta(preprocess_end_time, preprocess_start_time)
    #logger.debug(f'Preprocessing Template done in {delta_preprocess.hours} hrs, {delta_preprocess.minutes} mins, {delta_preprocess.seconds} sec')
#UNCOMMENT
    # Normalization Template
    #norm_start_time = datetime.datetime.now()
    #normalize(preprocessed_template_filepath, normalized_template_filepath)
    #norm_end_time = datetime.datetime.now()
    #delta_norm = relativedelta(norm_end_time, norm_start_time)
    #logger.debug(f'Normalization done in {delta_norm.hours} hrs, {delta_norm.minutes} mins, {delta_norm.seconds} sec')

    normed = os.listdir(normalized_uninjected_filepath)
    normed_injected = os.listdir(normalized_filepath)
    template_dir = os.listdir(normalized_template_filepath)
    print("start final")

    def stretch_template(template, stretch_factor):
        #stretch_factor=1
        # stretch_factor: (0, inf) real number to horizontally scale the template by
        # assumption: center of the template is the center to stretch by, template has odd length
        assert template.shape[0] % 2 == 1, 'Template must have odd number of points'
        k = template.shape[0] // 2
        orig_x = np.arange(-k, k+1)
        interp_func = interp1d(orig_x, template, bounds_error=False, fill_value=0)
        new_k = ((2*k + 1)/stretch_factor - 1)/2 # 2k_1 + 1 = sf*(2k_0 + 1)
        new_x = np.linspace(-new_k, new_k, template.shape[0])
        
        return interp_func(new_x)

    def get_a(target, template):
        return np.dot(target, template) / (np.dot(template, template) + 1e-12)
    def new_fom(freqs, spectrum, template, orig_freq, pad=False):
        # orig_freq: frequency at which the template was created
        a_vals = np.zeros(len(spectrum) - len(template))
        residuals = np.zeros(len(spectrum) - len(template))
        for start in range(len(spectrum) - len(template)):
            window = spectrum[start:start+len(template)]
            central_freq = freqs[start+len(window)//2]
            stretched = stretch_template(template, central_freq / orig_freq)
            a = get_a(window, stretched) 
            a_vals[start] = a 
            residuals[start] = np.sqrt(np.mean((a*stretched-window)**2)) 
        if pad:
            a_vals = np.pad(a_vals, len(stretched//2))
            residuals = np.pad(residuals, len(stretched)//2, mode='constant', constant_values=1)

        return a_vals / (residuals + 1e-12)


    std_cutoff = .004
    # std_cutoff = .25

    def std_filter():
        stds = []
        all_stds = []
        good_names = []
        print(len(normed))
        for name in normed:
            try:
                
                spec = np.load(os.path.join(normalized_uninjected_filepath, name))
                if np.std(spec) < std_cutoff:
                    if np.size(spec) == 4249:
                        good_names.append(name)
                        stds.append(np.std(spec))   
                all_stds.append(np.std(spec))

            except Exception as e:
                #logger.debug(e)
                pass
        
        return good_names

    # filtering by STD
    good_names = std_filter()
    # print(len(good_names))
    #logger.debug(f'number of spectra: {len(good_names)}')

    def asymmetry_builder(doppler_or_intensity):
        print(doppler_or_intensity)
        forward = []
        backward = []

        for i in good_names:
            try:
                #where are phi and theta in lband data
                theta = data_table.loc[i[:-4], 'sep90']
                phi = data_table.loc[i[:-4], 'sep_center']

                if doppler_or_intensity == 'doppler':
                    angle = theta
                    lower_cutoff = 69
                    upper_cutoff = 78
                elif doppler_or_intensity == 'intensity':
                    angle = phi
                    lower_cutoff = 45
                    upper_cutoff = 133

                if angle < lower_cutoff:
                    forward.append(i)
        
                elif angle > upper_cutoff:
                    backward.append(i)
            except Exception as e:
                #logger.debug(e)
                pass

        # loading template specs
        forward_specs_template = []
        backward_specs_template = []
        for name in template_dir:
            if name in forward:
                forward_specs_template.append(np.load(os.path.join(normalized_template_filepath, name)))
            elif name in backward:
                backward_specs_template.append(np.load(os.path.join(normalized_template_filepath, name)))
        #logger.debug(len(forward_specs_template), len(backward_specs_template))

        # loading uninjected specs
        forward_specs = []
        backward_specs = []
        for name in normed:
            if name in forward: #MAKE SURE THESE ARE FORMATED CORRECTLY
                forward_specs.append(np.load(os.path.join(normalized_uninjected_filepath, name)))
            elif name in backward:
                backward_specs.append(np.load(os.path.join(normalized_uninjected_filepath, name)))
        #logger.debug(len(forward_specs), len(backward_specs))

        # loading injected specs
        forward_specs_injected = []
        backward_specs_injected = []
        for name in normed_injected:
            if name in forward:
                forward_specs_injected.append(np.load(os.path.join(normalized_filepath, name)))
            elif name in backward:
                backward_specs_injected.append(np.load(os.path.join(normalized_filepath, name)))
        #logger.debug(len(forward_specs_injected), len(backward_specs_injected))
        # print(len(forward))
        # print(len(backward))
        # creating asymmetries
        forward_mean = np.mean(forward_specs, axis=0)
        backward_mean = np.mean(backward_specs, axis=0)
        forward_mean_injected = np.mean(forward_specs_injected, axis=0)
        backward_mean_injected = np.mean(backward_specs_injected, axis=0)
        forward_mean_template = np.mean(forward_specs_template, axis=0)
        backward_mean_template = np.mean(backward_specs_template, axis=0)

        asymmetry = (forward_mean-backward_mean)/(forward_mean+backward_mean)
        asymmetry_injected = (forward_mean_injected-backward_mean_injected)/(forward_mean_injected+backward_mean_injected)
        asymmetry_template = (forward_mean_template-backward_mean_template)/(forward_mean_template+backward_mean_template)
        # if doppler_or_intensity == 'doppler':
        #     np.save('/home/dataadmin/GBTData/SharedDataDirectory/lband/asymmetry_injected.npy', asymmetry_injected)
        
        # plt.plot(asymmetry_template)
        # plt.show()
        idx1 = np.argmin(np.abs(xs - 1030)) #based on where signal is injected
        idx2 = np.argmin(np.abs(xs - 1044))
        template = asymmetry_template[idx1:idx2-1]
        xs_fom = xs[len(template)//2:len(asymmetry_injected)-len(template)//2-1]

        fom = new_fom(xs, asymmetry_injected, template, 1037)
        fom_raw = new_fom(xs, asymmetry, template, 1037)
        return(xs_fom, fom_raw, fom)
         #return

    # need to fill this out!
    def whack_a_mole(fom_raw):
        # def exclude(start_freq, end_freq):
        #     idx1 = np.argmin(np.abs(xs_fom-start_freq))
        #     idx2= np.argmin(np.abs(xs_fom-end_freq))
        #     fom_raw[idx1:idx2] = 0
        # exclude(1925, 1926.5)
        # exclude(1997.5, 1999.5)
        # exclude(2039.5, 2041.5)
        # exclude(2069, 2071)
        # exclude(2073, 2074.5)
        # exclude(2213, 2215)
        # exclude(2222.5, 2224.5)
        # exclude(2311, 2313)
        # exclude(2338, 2340)
        # exclude(2453.5, 2455)
        # exclude(2472, 2474)
        # exclude(2482, 2484)
        # exclude(2515, 2517)
        # exclude(2555, 2557)
        # exclude(2125.5, 2127)
        # exclude(2535, 2536.5)
        return(fom_raw)


    def p_value_finder(fom_raw, fom):
        
        # calculate moving STD
        moving_std = []
        for i in range(len(fom_raw)):
            if i < 140:
                moving_std.append(np.std(fom_raw[:140]))
            elif i >= len(fom_raw) - 140:
                moving_std.append(np.std(fom_raw[len(fom_raw) - 140:]))
            else:
                moving_std.append(np.std(fom_raw[i-140:i+140]))


        idx = np.isfinite(fom) #all false need to be all true
        y_new = (fom/moving_std)[idx]
        q = [.5*(1-math.erf(y/np.sqrt(2))) for y in y_new]
        
        p_vals = np.log10(q)
    

        return p_vals

    # goes through spectrum and finds p_values of injected signals from the start frequency
    def signal_pvals(fom_xs, p_spec):
        
        p_vals = []
        for freq in range(start_frequency+50, int(fom_xs[-1]), 50):
            idx_center = np.argmin(np.abs(freq - fom_xs))
            p_val = np.min(p_spec[idx_center-10:idx_center+10])
            p_vals.append(p_val)
        return(p_vals)


    doppler_builder = asymmetry_builder('doppler')
    xs_fom = doppler_builder[0]
    doppler_pvals = p_value_finder(whack_a_mole(doppler_builder[1]), doppler_builder[2])
    intensity_builder = asymmetry_builder('intensity')
    intensity_pvals = p_value_finder(whack_a_mole(intensity_builder[1]), intensity_builder[2])
    combined_pvals = doppler_pvals + intensity_pvals
    p_vals_array = signal_pvals(xs_fom, combined_pvals)

    os.makedirs(filepath_to_save, exist_ok=True)
    np.save(filepath_to_save + '/doppler_pvals.npy', doppler_pvals)
    np.save(filepath_to_save + '/intensity_pvals.npy', intensity_pvals)
    np.save(filepath_to_save + '/combined_pvals.npy', combined_pvals)
    np.save(filepath_to_save + '/p_vals_array.npy', p_vals_array)
    np.save(filepath_to_save + '/xs_fom.npy', xs_fom)
    end_time = datetime.datetime.now()
    delta = relativedelta(end_time, start_time)
    logger.debug(f'Analysis done for {start_frequency} and {injected_signal} in {delta.hours} hrs, {delta.minutes} mins, {delta.seconds} sec total!') 
