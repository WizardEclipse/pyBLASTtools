import pyBLASTtools.detector as pbt
import numpy as np
import matplotlib.pyplot as plt
import os,sys

roach_num = int(sys.argv[1])
if roach_num == 1:
    array = "500"
    float_chans = 381
elif roach_num == 2:
    array = "250V"
    float_chans = 474
elif roach_num == 3:
    array = "350"
    float_chans = 667
elif roach_num == 4:
    array = "250U"
    float_chans = 498
elif roach_num == 5:
    array = "250W"
    float_chans = 511

flight_targ = np.load("flight_targ_rel_paths.npy")
flight_targ = flight_targ.astype("str")
flight_chop_ts = np.load("flight_ts_rel_paths.npy")
flight_chop_ts = flight_chop_ts.astype("str")

##################################3###############
# Change hard drive path
hard_drive_path = '/media/adrian/blast2020fc1/'
##################################################
channel_number = 40
cal_lamp = [32780000,32784869]
#home_wide = [38400000,38735000]
# get timestreams for all channels, start_samp and stop_samp are roach indices.
print("Loading timestreams..")
#det = pbt.kidsutils(data_dir = hard_drive_path + flight_chop_ts[roach_num-1], roach_num = roach_num, single = True, chan = channel_number, first_sample = cal_lamp[0], last_sample = cal_lamp[1])
det = pbt.kidsutils(data_dir = hard_drive_path + flight_chop_ts[roach_num-1], roach_num = roach_num, single = False, chan = float_chans, first_sample = cal_lamp[0], last_sample = cal_lamp[1])
print("Calculating delta f..")
df_x_all, df_y_all = det.get_df(path_to_sweep=hard_drive_path + flight_targ[roach_num-1],shift_idx=0)

