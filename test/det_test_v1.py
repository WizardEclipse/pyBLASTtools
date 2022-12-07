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

flight_chop_ts =['roach1_2020-01-06-06-22-01','roach2_2020-01-06-06-22-01', 'roach3_2020-01-06-06-21-56', 'roach4_2020-01-06-06-22-01', 'roach5_2020-01-06-06-22-01']

##################################3###############
# Change hard drive path
#hard_drive_path = '/data_storage_2/FLIGHT/flight_data/'
hard_drive_path = '/media/adrian/blast2020fc1/fc1/extracted/'
targ_drive_path = '/media/adrian/blast2020fc1/'
##################################################
channel_number = 101
cal_lamp = [32780000,32784869]
#home_wide = [38400000,38735000]
# get timestreams for all channels, start_samp and stop_samp are roach indices.
print("Loading timestreams..")
#det = pbt.kidsutils(data_dir = hard_drive_path + flight_chop_ts[roach_num-1], roach_num = roach_num, single = True, chan = channel_number, first_sample = cal_lamp[0], last_sample = cal_lamp[1])
det = pbt.kidsutils(data_dir = hard_drive_path + flight_chop_ts[roach_num-1], roach_num = roach_num, single = False, chan = float_chans, first_sample = cal_lamp[0], last_sample = cal_lamp[1])
print("Calculating delta f..")
df_x_all, df_y_all = det.get_df(path_to_sweep=targ_drive_path + flight_targ[roach_num-1],shift_idx=0)

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(df_x_all[channel_number]-np.mean(df_x_all[channel_number]),color='orange',label='$\Delta f$ Frequency')
ax2.plot(df_y_all[channel_number]-np.mean(df_y_all[channel_number]),color='turquoise',label='$\Delta f$ Dissipation')
ax1.set_ylabel("$\Delta f$ [Hz]")
ax2.set_ylabel("$\Delta f$ [Hz]")
ax1.set_ylim(-15000,15000)
ax2.set_ylim(-15000,15000)
ax1.set_title(array + " chan: "+str(channel_number))
ax1.legend()
ax2.legend()
plt.tight_layout()
plt.show()
