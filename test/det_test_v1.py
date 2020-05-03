import pyBLASTtools as pbt
import numpy as np
import matplotlib.pyplot as plt

roach_num = 2
if roach_num == 1:
    array = "500"
    float_chans = 380
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
    float_chans = 450

flight_targ = np.load("flight_targ.npy")
flight_targ = flight_targ.astype("str")
flight_chop_ts = np.load("flight_chop_ts.npy")
flight_chop_ts = flight_chop_ts.astype("str")

det = pbt.detector.kidsutils()

channel_number = 40
# get timestreams for specific channel
I_chan, Q_chan = det.getAllTs(flight_chop_ts[roach_num-1],roach_num,float_chans,start_samp = 32780000, stop_samp=32780000+4869)
Z = I_chan + 1j*Q_chan
# get target sweep for all channels
s21_real, s21_imag = det.loadBinarySweepData(flight_targ[roach_num-1],vna=False)
s21_real_f, s21_imag_f = det.despike_targs(s21_real.T, s21_imag.T)
s21_f = s21_real_f + 1j*s21_imag_f
# get df's
df_x_all, df_y_all = det.get_all_df_gradients( s21_f, Z, 2)
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
