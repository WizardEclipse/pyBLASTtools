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

conversion = pbt.pointing.utils(az=az_nodrift, el=pointing_data_resample['MC_EL_MOTOR_POS'], \
                                radec_frame='current',\
                                lon=pointing_data_resample['LON'], \
                                lat=pointing_data_resample['LAT'], \
                                height=pointing_data_resample['ALT'], \
                                time=d.time_master)

radec = conversion.horizontal2sky()
pa = conversion.parallactic_angle()

t = pbt.pointing.convert_to_telescope(radec[0], radec[1], pa=pa)
radec = t.conversion()

pa = d.interpolate(master_array=pa, fs_master=100)