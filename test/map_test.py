import pyBLASTtools as pbt
import glob as gl 
from itertools import compress
import pygetdata as gd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sgn

from astropy.time import Time

path = '/media/gabriele/mac'

list_file = gl.glob(path+'/*')

master_file = list_file['master' in list_file]
roach_file = list(compress(list_file, ['roach' in s for s in list_file]))

### Scan Parameters ###
time_start = 1578323354+60*4.2
time_end = 1578324356+60*3

d = pbt.timing.dirfile_interp('time_val', path_master=master_file, path_roach=roach_file[0], \
                              time_start=time_start, time_end=time_end, roach_num='1', offset=0)

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

det = pbt.detector.kidsutils(data_dir=roach_file[0], roach_num=1, single=True, chan=10, \
                             first_sample=d.idx_start_roach, last_sample=d.idx_end_roach)

phase = det.phase

phase_trend = pbt.detector.detector_trend(phase)
phase_detrend, baseline = phase_trend.fit_residual(baseline=True, return_baseline=True, \
                                                   baseline_type='poly', order=8, tol=5e-4)

ctype = 'RA and DEC'
crpix = np.array([50.,50.])
crval = np.array([np.median(radec[0]), np.median(radec[1])])
pixel_size = np.array([25./3600., 25./3600.])

wcs = pbt.mapmaker.wcs_world(crpix=crpix, cdelt=pixel_size, crval=crval, telcoord=True)
wcs.wcs_proj(radec[0], radec[1], 1)

proj = wcs.w.copy()
w = wcs.pixel.copy()

maps=pbt.mapmaker.mapmaking(phase_detrend, 1., np.zeros(len(phase_detrend)), w, crpix)

maps = maps.binning_map()