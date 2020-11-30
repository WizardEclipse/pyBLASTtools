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

#flight_targ = np.load("flight_targ_rel_paths.npy")
#flight_targ = flight_targ.astype("str")
#flight_chop_ts = np.load("flight_ts_rel_paths.npy")
#flight_chop_ts = flight_chop_ts.astype("str")

flight_chop_ts =['roach1_2020-01-06-06-22-01','roach2_2020-01-06-06-22-01', 'roach3_2020-01-06-06-21-56', 'roach4_2020-01-06-06-22-01', 'roach5_2020-01-06-06-22-01']

##################################3###############
# Change hard drive path
# hard_drive_path = '/media/adrian/blast2020fc1/'
hard_drive_path = '/data_storage_2/FLIGHT/flight_data/'
##################################################
<<<<<<< Updated upstream
channel_number = 40
cal_lamp = [32780000,32784869]
#home_wide = [38400000,38735000]
# get timestreams for all channels, start_samp and stop_samp are roach indices.
print("Loading timestreams..")
#det = pbt.kidsutils(data_dir = hard_drive_path + flight_chop_ts[roach_num-1], roach_num = roach_num, single = True, chan = channel_number, first_sample = cal_lamp[0], last_sample = cal_lamp[1])
det = pbt.kidsutils(data_dir = hard_drive_path + flight_chop_ts[roach_num-1], roach_num = roach_num, single = False, chan = float_chans, first_sample = cal_lamp[0], last_sample = cal_lamp[1])
print("Calculating delta f..")
df_x_all, df_y_all = det.get_df(path_to_sweep=hard_drive_path + flight_targ[roach_num-1],shift_idx=0)

=======
channel_number = 74

det = pbt.kidsutils(data_dir = hard_drive_path + flight_chop_ts[roach_num -1], roach_num = roach_num, single = False, chan = float_chans, first_sample = 42000000, last_sample = 45000000)

phase = np.arctan2(det.Z.imag[channel_number], det.Z.real[channel_number])

sampling_rate = 512.0e6/2**(20) # Hz
sampling_period = 1./sampling_rate
time = np.linspace(0,len(phase)-1,len(phase))*sampling_period
plt.figure()

import os
os.chdir("/home/anaconda/pyBLASTtools/test/")
np.save("Time.npy", time)
np.save("phase.npy", phase)

plt.plot(time, phase)
plt.ylabel("Phase [rad]")
plt.xlabel("Time [s]")
plt.show()
"""
# get target sweep for all channels
s21_real, s21_imag = det.loadBinarySweepData(hard_drive_path + flight_targ[roach_num-1],vna=False)
#s21_real_f, s21_imag_f = det.despike_targs(s21_real.T, s21_imag.T)
s21_f = s21_real + 1j*s21_imag

# get df's
df_x_all, df_y_all = det.get_df( s21_f = s21_f, timestream = det.Z)
#df_x_all, df_y_all = det.get_all_df_gradients( s21_f, Z, 2)
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(df_x_all[channel_number]-np.mean(df_x_all[channel_number]),color='orange',label='$\Delta f$ Frequency')
ax2.plot(df_y_all[channel_number]-np.mean(df_y_all[channel_number]),color='turquoise',label='$\Delta f$ Dissipation')
ax1.set_ylabel("$\Delta f$ [Hz]")
ax2.set_ylabel("$\Delta f$ [Hz]")
ax1.set_title(array + " chan: "+str(channel_number))
ax1.legend()
ax2.legend()
plt.tight_layout()
plt.show()
"""
>>>>>>> Stashed changes
