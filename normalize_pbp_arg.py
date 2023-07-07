from copy import deepcopy
import normalize_one_pbp
import sys, subprocess, os
import multiprocessing as mp
import numpy as np

sys.path.insert(0, '..')

#main_file_path = '/home/dataadmin/GBTData/SharedDataDirectory/sband/preprocessed_limits_analysis/start_1800/signal_1e-43'
a = 1
b = 1
order = 3
window = 55


def normalize(main_file_path, file_path_to_save):
    os.makedirs(file_path_to_save, exist_ok=True)
    files = os.listdir(main_file_path); 
    num_workers = mp.cpu_count()
    #print(num_workers)
    pool = mp.Pool(num_workers)
    conf = { 
        'window': window, 
        'filepath': '', 
        'out_filepath' : '',
        'worker_id': -1, 
    } 
    # import pdb; 
    print(len(files))
    for i in range(0, len(files), num_workers): 
        for j in range(1, num_workers+1): 
            if i + j >= len(files): 
                break 
            conf['worker_id'] = i + j 
            conf['filepath'] = os.path.join(main_file_path, files[i + j])
            conf['out_filepath'] = os.path.join(file_path_to_save, files[i+ j])
            passable_dict = deepcopy(conf)
            # pdb.set_trace()
            pool.apply_async(normalize_one_pbp.main, kwds=passable_dict)
            
    
    pool.close()
    pool.join()
