from pval_finder import main_function
import os
import sys 
import configparser
from tqdm import tqdm

def main(start_frequencies, signal_sizes, config): 
    os.makedirs("PValRunnerLogs", exist_ok=True)

    for freq in start_frequencies:
        for signal in signal_sizes:
            main_function(freq, signal, config)

if __name__ == '__main__': 
    config = configparser.ConfigParser()
    config.read('config.ini')
    start_frequencies = [int(arg) for arg in config.get("Settings", "start_frequencies").split(",")]
    signal_sizes = [float(arg) for arg in config.get("Settings", "signal_sizes").split(",")]
    main(start_frequencies, signal_sizes, config)
