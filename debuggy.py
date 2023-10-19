import pyBLASTtools as pbt
import pygetdata as gd
import glob as gl 
from itertools import compress
import matplotlib.pyplot as plt
import numpy as np
import os
from astropy.table import Table
from astropy.coordinates import EarthLocation, solar_system_ephemeris, get_body
import astropy.units as u
from astropy.time import Time
from pyBLASTtools import timing as tm

def create_ctimefile(path, roach_path, roachnum = 1):
    t = tm.timing(path=path, roach_path=roach_path)
    
    # master = t.ctime_master(write=True)
    # ctime_master = t.time_master
    
    roach_number = roachnum #, 2, 3, 4, 5]
    
    kind = ['Packet', 'Clock']
    
    roach_comparison = {}
    
    roach = t.ctime_roach(roach_number, kind, mode='average', write=True)

path = '/media/triv/blast2020fc1/fc1/extracted'

list_file = gl.glob(path+'/*')

master_file = [s for s in list_file  if 'master'  in s]
master_file = master_file[0]
roach_file = list(compress(list_file, ['roach' in s for s in list_file]))

### Detector Parameters ###
roach_number = 3
detector_number = int(35)
roach_file_path = path+'/roach3_2020-01-06-06-21-56'

roach_file_path

### Scan Parameters ###
time_start = 1578323354+60*4.2
time_end = 1578324356+60*3
time_offset = 0.

### Mask for cal_lamp ###
idx_mask_start = int(np.floor(408400+time_offset/488.28125))
idx_mask_end = int(np.ceil(409800+time_offset/488.28125))

### Mars roach index ###
idx_start = 44771610
idx_end = 45686500
#idx_start = 38213478
#idx_end = 38702626

create_ctimefile(path='/media/triv/blast2020fc1/fc1/extracted', roach_path=None, roachnum=[3])


