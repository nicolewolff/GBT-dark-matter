from copy import deepcopy
import preprocess_one 
import sys, subprocess, os
import multiprocessing as mp
import numpy as np

sys.path.insert(0, '..')

#what we want to preprocess
#file_path_to_extract = '/home/dataadmin/GBTData/SharedDataDirectory/sband/injected_limits_analysis/start_1800/signal_1e-43'

def preprocess(file_path_to_extract, file_path_to_save, template): 
    os.makedirs(file_path_to_save, exist_ok=True)
    files = os.listdir(file_path_to_extract); 
    num_workers = mp.cpu_count()
    pool = mp.Pool(num_workers)
    conf = {   
        'filepath': '', 
        'out_filepath' : '',
        'worker_id': -1, 
        'template': False, 
    } 
    # import pdb; 
    for i in range(0, len(files), num_workers): 
        for j in range(1, num_workers+1): 
            if i + j >= len(files): 
                break 
            conf['worker_id'] = i + j 
            conf['filepath'] = os.path.join(file_path_to_extract, files[i + j])
            conf['template'] = template
            conf['out_filepath'] = os.path.join(file_path_to_save, files[i+ j])
            passable_dict = deepcopy(conf)
            # pdb.set_trace()
            pool.apply_async(preprocess_one.main, kwds=passable_dict)
    
    pool.close()
    pool.join()
    

