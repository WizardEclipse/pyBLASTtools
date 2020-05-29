import pyBLASTtools as pbt
import glob as gl 
from itertools import compress
import matplotlib.pyplot as plt
import numpy as np
import os

path = '/media/gabriele/mac'

list_file = gl.glob(path+'/*')

master_file = [s for s in list_file  if 'master'  in s]
master_file = master_file[0]
roach_file = list(compress(list_file, ['roach' in s for s in list_file]))

### Detector Parameters ###
roach_number = 1
detector_number = 2

### Scan Parameters ###
time_start = 1578323354+60*4.2
time_end = 1578324356+60*3
time_offset = 0.

### Mask for cal_lamp ###
idx_mask_start = int(np.floor(408400+time_offset/488.28125))
idx_mask_end = int(np.ceil(409800+time_offset/488.28125))

d = pbt.timing.dirfile_interp('time_val', path_master=master_file, path_roach=roach_file[int(roach_number-1)], \
                              time_start=time_start, time_end=time_end, roach_num=str(int(roach_number)), \
                              offset=time_offset)

field_list = ['AZ', 'MC_EL_MOTOR_POS', 'AZ_RAW_DGPS', 'LON', 'LAT', 'ALT']

pointing_data = pbt.data.data(master_file, d.idx_start_master, d.idx_end_master, field_list, mode='samples')
pointing_data.resample()
pointing_data_resample = pointing_data.data_values

pointing_data_resample['AZ_RAW_DGPS'] += 270.90
pointing_data_resample['MC_EL_MOTOR_POS'] += -0.072
pointing_data_resample['AZ_RAW_DGPS'] -= 360.

az_nodrift = pbt.utils.remove_drift(pointing_data_resample['AZ'], \
                                    pointing_data_resample['AZ_RAW_DGPS'], 100, 100)

conversion = pbt.pointing.utils(az=az_nodrift, alt=pointing_data_resample['MC_EL_MOTOR_POS'], \
                                radec_frame='current',\
                                lon=pointing_data_resample['LON'], \
                                lat=pointing_data_resample['LAT'], \
                                height=pointing_data_resample['ALT'], \
                                time=d.time_master)

radec = conversion.horizontal2sky()
pa = conversion.parallactic_angle()

ra = d.interpolate(master_array=radec[0], fs_master=100)
dec = d.interpolate(master_array=radec[1], fs_master=100)
pa = d.interpolate(master_array=pa, fs_master=100)

t = pbt.pointing.convert_to_telescope(ra, dec, pa=pa)
radec = t.conversion()

det = pbt.detector.kidsutils(data_dir=roach_file[int(roach_number-1)], roach_num=int(roach_number), \
                             single=False, chan=int(detector_number), first_sample=d.idx_start_roach, \
                             last_sample=d.idx_end_roach)

path_file = os.path.dirname(os.path.abspath(__file__))
flight_targ = np.load(path_file+'/flight_targ_rel_paths.npy')
flight_targ = flight_targ.astype("str")

df_x, df_y = det.get_df(path_to_sweep=path+'/'+flight_targ[roach_number-1])

sys.exit()

# phase = det.phase

# mask = np.zeros_like(phase[0], dtype=bool)
# mask[idx_mask_start:idx_mask_end] = True

# phase = np.ma.array(phase, mask=np.tile(mask, (np.shape(phase)[0], 1)))

# phase_trend = pbt.detector.detector_trend(phase)
# phase_detrend, baseline = phase_trend.fit_residual(baseline=True, return_baseline=True, \
#                                                    baseline_type='poly', order=8, tol=5e-4)

# final_ra = radec[0][:,~mask]
# final_dec = radec[1][:,~mask]
# final_det = phase_detrend.data[:,~mask]

ctype = 'RA and DEC'
crpix = np.array([50.,50.])
crval = np.array([np.median(radec[0]), np.median(radec[1])])
pixel_size = np.array([35./3600., 35./3600.])

wcs = pbt.mapmaker.wcs_world(crpix=crpix, cdelt=pixel_size, crval=crval, telcoord=True)
wcs.wcs_proj(final_ra, final_dec, np.shape(final_det)[0])

proj = wcs.w.copy()
w = wcs.pixel.copy()

maps=pbt.mapmaker.mapmaking(final_det, 1., np.zeros(len(phase_detrend)), w, crpix)

maps = maps.binning_map(coadd=True)

print(maps['I'])