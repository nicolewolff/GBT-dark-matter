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
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger()

def main_function(start_frequency, injected_signal, config):
    logger.debug(f"Starting process for {start_frequency} and {injected_signal}")
    start_time = datetime.datetime.now()
    injected_filepath = config.get("Paths", "injected_filepath").format(start_frequency=start_frequency, injected_signal=injected_signal)
    uninjected_filepath = config.get("Paths", "uninjected_filepath").format(start_frequency=start_frequency, injected_signal=injected_signal)    
    preprocessed_filepath = config.get("Paths", "preprocessed_filepath").format(start_frequency=start_frequency, injected_signal=injected_signal)
    preprocessed_uninjected_filepath = config.get("Paths", "preprocessed_uninjected_filepath").format(start_frequency=start_frequency, injected_signal=injected_signal)
    normalized_filepath = config.get("Paths", "normalized_filepath").format(start_frequency=start_frequency, injected_signal=injected_signal)
    normalized_uninjected_filepath = config.get("Paths", "normalized_uninjected_filepath")
    template_filepath = config.get("Paths", "template_filepath")
    data_table_filepath = config.get("Paths", "data_table_filepath")
    xs_filepath = config.get("Paths", "xs_filepath")
    filepath_to_save = config.get("Paths", "filepath_to_save").format(start_frequency=start_frequency, injected_signal=injected_signal)
    normalized_template_filepath = config.get("Paths", "normalized_template_filepath")
    injected_template_filepath = config.get("Paths", "injected_template_filepath")
    preprocessed_template_filepath = config.get("Paths", "preprocessed_template_filepath")
    data_table = pd.read_csv(data_table_filepath).set_index('file_name')
    xs = np.load(xs_filepath)

    if config.getboolean("Operations", "inject"):
        inject_start_time = datetime.datetime.now()
        inject_spaced(injected_signal, start_frequency, injected_filepath, False)
        inject_end_time = datetime.datetime.now()
        delta_inject = relativedelta(inject_end_time, inject_start_time)
        logger.debug(f'Injection done in {delta_inject.hours} hrs, {delta_inject.minutes} mins, {delta_inject.seconds} sec')
        print("done inject")

    if config.getboolean("Operations", "preprocess"):
        preprocess_start_time = datetime.datetime.now()
        print("About to preprocess:")
        preprocess(uninjected_filepath, preprocessed_filepath, False)
        preprocess_end_time = datetime.datetime.now()
        delta_preprocess = relativedelta(preprocess_end_time, preprocess_start_time)
        logger.debug(f'Preprocessing done in {delta_preprocess.hours} hrs, {delta_preprocess.minutes} mins, {delta_preprocess.seconds} sec')
        print("done preprocess")
    
    if config.getboolean("Operations", "uninjected_preprocess"):
        preprocess_start_time = datetime.datetime.now()
        print("About to preprocess:")
        preprocess(uninjected_filepath, preprocessed_uninjected_filepath, False)
        preprocess_end_time = datetime.datetime.now()
        delta_preprocess = relativedelta(preprocess_end_time, preprocess_start_time)
        logger.debug(f'Preprocessing done in {delta_preprocess.hours} hrs, {delta_preprocess.minutes} mins, {delta_preprocess.seconds} sec')
        print("done uninject preprocess")

    if config.getboolean("Operations", "normalize"):
        norm_start_time = datetime.datetime.now()
        print('starting norm')
        normalize(preprocessed_filepath, normalized_filepath)
        norm_end_time = datetime.datetime.now()
        delta_norm = relativedelta(norm_end_time, norm_start_time)
        logger.debug(f'Normalization done in {delta_norm.hours} hrs, {delta_norm.minutes} mins, {delta_norm.seconds} sec')
        print("done norm")

    if config.getboolean("Operations", "uninjected_normalize"):
        norm_start_time = datetime.datetime.now()
        normalize(preprocessed_uninjected_filepath, normalized_uninjected_filepath)
        norm_end_time = datetime.datetime.now()
        delta_norm = relativedelta(norm_end_time, norm_start_time)
        logger.debug(f'Uninjected normalization done in {delta_norm.hours} hrs, {delta_norm.minutes} mins, {delta_norm.seconds} sec')
        print("done uninject norm")

    if config.getboolean("Operations", "inject_template"): 
        inject_start_time = datetime.datetime.now()
        inject_spaced(injected_signal, start_frequency, injected_template_filepath, True)
        inject_end_time = datetime.datetime.now()
        delta_inject = relativedelta(inject_end_time, inject_start_time)
        logger.debug(f'Injection Template done in {delta_inject.hours} hrs, {delta_inject.minutes} mins, {delta_inject.seconds} sec')

    if config.getboolean("Operations", "preprocess_template"):
        preprocess_start_time = datetime.datetime.now()
        preprocess(injected_template_filepath, preprocessed_template_filepath, True)
        preprocess_end_time = datetime.datetime.now()
        delta_preprocess = relativedelta(preprocess_end_time, preprocess_start_time)
        logger.debug(f'Preprocessing Template done in {delta_preprocess.hours} hrs, {delta_preprocess.minutes} mins, {delta_preprocess.seconds} sec')

    if config.getboolean("Operations", "normalize_template"):
        norm_start_time = datetime.datetime.now()
        normalize(preprocessed_template_filepath, normalized_template_filepath)
        norm_end_time = datetime.datetime.now()
        delta_norm = relativedelta(norm_end_time, norm_start_time)
        logger.debug(f'Normalization done in {delta_norm.hours} hrs, {delta_norm.minutes} mins, {delta_norm.seconds} sec')

    def stretch_template(template, stretch_factor):
        # Horizontally scale the template
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
        # Figure of Merit
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
                pass
        
        return good_names

    def asymmetry_builder(doppler_or_intensity):
        print(doppler_or_intensity)
        forward = []
        backward = []

        for i in good_names:
            try:
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
                pass

        forward_specs_template = []
        backward_specs_template = []
        for name in template_dir:
            if name in forward:
                forward_specs_template.append(np.load(os.path.join(normalized_template_filepath, name)))
            elif name in backward:
                backward_specs_template.append(np.load(os.path.join(normalized_template_filepath, name)))

        forward_specs = []
        backward_specs = []
        for name in normed:
            if name in forward:
                forward_specs.append(np.load(os.path.join(normalized_uninjected_filepath, name)))
            elif name in backward:
                backward_specs.append(np.load(os.path.join(normalized_uninjected_filepath, name)))

        # Loading injected spectra
        forward_specs_injected = []
        backward_specs_injected = []
        for name in normed_injected:
            if name in forward:
                forward_specs_injected.append(np.load(os.path.join(normalized_filepath, name)))
            elif name in backward:
                backward_specs_injected.append(np.load(os.path.join(normalized_filepath, name)))

        # Forming asymmetries
        forward_mean = np.mean(forward_specs, axis=0)
        backward_mean = np.mean(backward_specs, axis=0)
        forward_mean_injected = np.mean(forward_specs_injected, axis=0)
        backward_mean_injected = np.mean(backward_specs_injected, axis=0)
        forward_mean_template = np.mean(forward_specs_template, axis=0)
        backward_mean_template = np.mean(backward_specs_template, axis=0)

        asymmetry = (forward_mean-backward_mean)/(forward_mean+backward_mean)
        asymmetry_injected = (forward_mean_injected-backward_mean_injected)/(forward_mean_injected+backward_mean_injected)
        asymmetry_template = (forward_mean_template-backward_mean_template)/(forward_mean_template+backward_mean_template)

        idx1 = np.argmin(np.abs(xs - 1030)) #based on where signal is injected
        idx2 = np.argmin(np.abs(xs - 1044))
        template = asymmetry_template[idx1:idx2-1]
        xs_fom = xs[len(template)//2:len(asymmetry_injected)-len(template)//2-1]

        fom = new_fom(xs, asymmetry_injected, template, 1037)
        fom_raw = new_fom(xs, asymmetry, template, 1037)
        return(xs_fom, fom_raw, fom)

    def whack_a_mole(fom_raw):
        def exclude(start_freq, end_freq):
             idx1 = np.argmin(np.abs(xs_fom-start_freq))
             idx2= np.argmin(np.abs(xs_fom-end_freq))
             fom_raw[idx1:idx2] = 0
        # Select noisy regions to exclude 
        return(fom_raw)


    def p_value_finder(fom_raw, fom):
        
        # Calculate moving Standard Deviation
        moving_std = []
        for i in range(len(fom_raw)):
            if i < 140:
                moving_std.append(np.std(fom_raw[:140]))
            elif i >= len(fom_raw) - 140:
                moving_std.append(np.std(fom_raw[len(fom_raw) - 140:]))
            else:
                moving_std.append(np.std(fom_raw[i-140:i+140]))


        idx = np.isfinite(fom)
        y_new = (fom/moving_std)[idx]
        q = [.5*(1-math.erf(y/np.sqrt(2))) for y in y_new]
        
        p_vals = np.log10(q)
    

        return p_vals

    def signal_pvals(fom_xs, p_spec):
        
        p_vals = []
        for freq in range(start_frequency+50, int(fom_xs[-1]), 50):
            idx_center = np.argmin(np.abs(freq - fom_xs))
            p_val = np.min(p_spec[idx_center-10:idx_center+10])
            p_vals.append(p_val)
        return(p_vals)

    if config.getboolean("Operations", "asymmetry"):
        normed = os.listdir(normalized_uninjected_filepath)
        normed_injected = os.listdir(normalized_filepath)
        template_dir = os.listdir(normalized_template_filepath)
        print("Starting asymmetry")

        # Filtering files with a Standard deviation cut
        good_names = std_filter()

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
