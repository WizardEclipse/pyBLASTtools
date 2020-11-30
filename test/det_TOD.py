import pyBLASTtools as pbt
import glob as gl 
from itertools import compress
import matplotlib.pyplot as plt
import numpy as np
import os

#path
path = '/data_storage_2/FLIGHT/flight_data'

list_file = gl.glob(path+'/*')

master_file = [s for s in list_file  if 'master'  in s]
master_file = master_file[0]
roach_file = list(compress(list_file, ['roach' in s for s in list_file]))

### Detector Parameters ###
roach_number = 3
detector_number = 454#np.arange(0,10, 1, dtype=int)
roach_file_path = path+'/roach3_2020-01-06-06-21-56'

### Scan Parameters ###
# time_start = 1578323354+60*-5
# time_end = 1578324356+60*10
# time_offset = 0.04

# ### Mask for cal_lamp ###
# idx_mask_start = int(np.floor(408400+time_offset/488.28125))
# idx_mask_end = int(np.ceil(409800+time_offset/488.28125))

### Index start for Mars Scan Roaches ###
start_idx = 44771610
stop_idx = 45686500

# dt_shift = 209634 + 40000
# dt_start = 20000
# start_idx = 32813043-dt_shift+dt_start
# stop_idx  = 33022677-dt_shift

det = pbt.detector.kidsutils(data_dir=roach_file_path, roach_num=int(roach_number), \
                             single=True, chan=detector_number, first_sample=start_idx, \
                             last_sample=stop_idx)

### Scan Parameters ###
# time_start = 1578323354+60*-5
# time_end = 1578324356+60*10
# time_offset = 0.04

# time_start = 1578323354+60*-4
# time_end = 1578324356+60*3
# time_offset = 0.04

# ### Mask for cal_lamp ###
# idx_mask_start = int(np.floor(414250+time_offset/488.28125))
# idx_mask_end = int(np.ceil(415750+time_offset/488.28125))

d = pbt.timing.dirfile_interp('idx_roach', path_master=master_file, path_roach=roach_file_path, \
                              idx_start=start_idx, idx_end=stop_idx, roach_num=str(int(roach_number)))

field_list = ['AZ']

pointing_data = pbt.data.data(master_file, d.idx_start_master, d.idx_end_master, field_list, mode='samples')
pointing_data.resample()
pointing_data_resample = pointing_data.data_values

az = pointing_data_resample['AZ']
az = d.interpolate(master_array=az, fs_master=100)

# det = pbt.detector.kidsutils(data_dir=roach_file_path, roach_num=int(roach_number), \
#                              single=True, chan=detector_number, first_sample=d.idx_start_roach, \
#                              last_sample=d.idx_end_roach)

path_file = os.path.dirname(os.path.abspath(__file__))
flight_targ = np.load(path_file+'/flight_targ_rel_paths.npy')
flight_targ = flight_targ.astype("str")

data, df_y = det.get_df(path_to_sweep=path+'/'+flight_targ[roach_number-1], window=20)

desp = pbt.detector.despike(data, thres=None, hthres=1, width=30)
# df_x_cleaned = desp.replace_peak()

trend = pbt.detector.detector_trend(data)
df_x_cleaned, flags, baseline = trend.fit_residual(baseline=True, return_baseline=True, \
                                                   baseline_type='poly', iter_val=10, order=1, tol=5e-4,
                                                   flag_nosignal=True)

phase = det.KIDphase()


plt.plot(np.arange(len(data[0])), data[0])
plt.plot(desp.idx_peak[0], data[0][desp.idx_peak[0]], 'x')
plt.plot(np.arange(len(data[0])), baseline[0])
plt.show()

plt.plot(az[desp.idx_peak[0]], data[0][desp.idx_peak[0]])
plt.show()

sys.exit()

plt.figure(0)
out = data[0][205600:207600]
plt.plot(np.arange(len(out))*0.55, out)
plt.xlabel('Yaw (arcsec)')
plt.ylabel('df')
plt.title('Out of Focus')

plt.figure(1)
out = data[0][381600:383600]
plt.plot(np.arange(len(out))*0.55, out)
plt.xlabel('Yaw (arcsec)')
plt.ylabel('df')
plt.title('In Focus')

plt.show()

# path_file = os.path.dirname(os.path.abspath(__file__))
# flight_targ = np.load(path_file+'/flight_targ_rel_paths.npy')
# flight_targ = flight_targ.astype("str")

# df_x, df_y = det.get_df(path_to_sweep=path+'/'+flight_targ[roach_number-1], window=20)


# desp = pbt.detector.despike(phase, thres=4)
# phase_cleaned = desp.replace_peak()

