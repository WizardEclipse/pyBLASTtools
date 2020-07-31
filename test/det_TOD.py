import pyBLASTtools as pbt
import glob as gl 
from itertools import compress
import matplotlib.pyplot as plt
import numpy as np
import os
from astropy.table import Table

path = '/data_storage_2/FLIGHT/flight_data'

list_file = gl.glob(path+'/*')

master_file = [s for s in list_file  if 'master'  in s]
master_file = master_file[0]
roach_file = list(compress(list_file, ['roach' in s for s in list_file]))

### Detector Parameters ###
roach_number = 3
detector_number = np.arange(0,668, 1, dtype=int)
roach_file_path = path+'/roach3_2020-01-06-06-21-56'

### Scan Parameters ###
time_start = 1578323354+60*4.2
time_end = 1578324356+60*3
time_offset = 0.

### Mask for cal_lamp ###
idx_mask_start = int(np.floor(408400+time_offset/488.28125))
idx_mask_end = int(np.ceil(409800+time_offset/488.28125))

### Index start for Mars Scan Roaches ###
idx_start = 44771610
idx_end = 45686500

dt_shift = 209634 + 40000
dt_start = 20000
start_idx = 32813043-dt_shift+dt_start
stop_idx  = 33022677-dt_shift

det = pbt.detector.kidsutils(data_dir=roach_file_path, roach_num=int(roach_number), \
                             single=False, chan=detector_number, first_sample=start_idx, \
                             last_sample=stop_idx)

phase = det.KIDphase()

path_file = os.path.dirname(os.path.abspath(__file__))
flight_targ = np.load(path_file+'/flight_targ_rel_paths.npy')
flight_targ = flight_targ.astype("str")

df_x_all, df_y_all = det.get_df(path_to_sweep=path+'/'+flight_targ[roach_number-1], window=20)

