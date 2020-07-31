import numpy as np
import scipy.signal as sgn
import pygetdata as gd
import sys
import os
from scipy import interpolate

class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class InputError(Error):
    """Exception raised for errors in the input. """
    pass
        
class despike:

    '''
    Class to despike the TOD
    '''

    def __init__(self, data, remove_mean=True, mean_std_distance=0.3, hthres=4, thres=4, pthres=None, \
                 width=np.array([1,10]), full_width=False, rel_height=0.5):

        '''
        These are the parameters for this class:

        - data: signal that needs to be analyzed
        - remove_mean: a parameter to choose if the mean is removed from the signal that needs 
                       to be analyzed just for the purpose of searching peaks
        - mean_std_distance: a parameter used to compute the useful data for estimating the mean 
                             and std of the signal. 
                             This parameter is given by (data-mean(data))/np.mean(data)
        - hthres: height of the peaks in unit of std
        - thres: vertical distance to neighbouring samples in unit of std
        - pthres: prominance of the peaks in unit of std 
        - width: required width of the peak to be found. If a single number that is the minimum 
                 width of the peak. If a 2D array the first element is the minimum width and the 
                 second one is the maximum width. If None, a default value of a maximum value of 
                 200 samples will be used for the width of the peak

        For hthres, thres and pthres more information can be found on the at scipy.signal.find_peaks
        '''

        self.data = data

        self.remove_mean = remove_mean

        if mean_std_distance < 1e-5 or mean_std_distance > 1:
            mean_std_distance = 0.3

        val = (data-np.mean(data, axis=1))/np.mean(data, axis=1)
        
        self.signal_mean = np.zeros(np.shape(data)[0])
        self.signal_std = np.zeros(np.shape(data)[0])

        for i in range(np.shape(self.data)[0]):
            self.signal_std[i] = np.std(self.data[i,val[i]<mean_std_distance], axis=1)
            self.signal_mean[i] = np.mean(self.data[i,val[i]<mean_std_distance], axis=1)
        
        if remove_mean:
            self.signal = data-self.signal_mean
        else:
            self.signal = data

        if hthres is not None:
            self.hthres = hthres*self.signal_std
        else:
            self.hthres = hthres

        if pthres is not None:
            self.pthres = pthres*self.signal_std
        else:
            self.pthres = pthres

        if thres is not None:
            self.thres = thres*self.signal_std
        else:
            self.thres = thres
        
        if width is not None:
            self.width = width
        else:
            self.width = np.array([1,200])

        self.idx_peak = []
        self.param_peak = []

        for i in range(np.shape(self.signal)[0]):

            peak_temp, param_temp = sgn.find_peaks(np.abs(self.signal[i]), height=self.hthres, \
                                                   prominence=self.pthres, \
                                                   threshold=self.thres, width=self.width, \
                                                   rel_height=rel_height)
            
            self.idx_peak.append(peak_temp)
            self.param_peak.append(param_temp)

        if full_width:

            for i in range(np.shape(self.signal)[0]):
                param_temp = sgn.peak_widths(np.abs(self.signal[i]), peaks=self.idx_peak, rel_height=1.)

                ledge = np.append(param_temp[-2][0], np.maximum(param_temp[-2][1:], param_temp[-1][:-1]))
                redge = np.append(np.minimum(param_temp[-2][1:], param_temp[-1][:-1]), param_temp[-1][-1])

                self.param_peak[i]['widths'] = param_temp[0]
                self.param_peak[i]['left_ips'] = ledge
                self.param_peak[i]['right_ips'] = redge

        for i in range(np.shape(self.signal)[0]):
            self.param_peak[i]['left_ips'] = (np.floor(self.param_peak[i]['left_ips'])).astype(int)
            self.param_peak[i]['right_ips'] = (np.floor(self.param_peak[i]['right_ips'])).astype(int)

    def replace_peak(self, peaks=None, ledge=None, redge=None, window=1000):

        '''
        This function replaces the spikes data with noise realization. Noise can be gaussian
        or poissonian based on the statistic of the data
        - peaks: indices of the peaks 
        - ledge: left edge of the peak width
        - redge: right edge of the peak width
        - window: dimension in samples of the data to be analyzed before and after the peak to 
                  estimate the mean, std and var 
        '''

        x_inter = np.array([], dtype = 'int')

        replaced = self.data.copy()

        for j in range(np.shape(self.data)[0]):
            if peaks is None:
                peaks = self.idx_peak[j]

            if ledge is None:
                ledge = self.param_peak[j]['left_ips']
            else:
                if isinstance(ledge, np.ndarray):
                    if len(ledge) != len(peaks):
                        ledge = peaks-np.amax(ledge)
                    else:
                        ledge = ledge
                else:
                    ledge = peaks-np.amax(ledge)            

            if redge is None:
                redge = self.param_peak[j]['right_ips']
            else:
                if isinstance(redge, np.ndarray):
                    if len(redge) != len(peaks):
                        redge = peaks+np.amax(redge)
                    else:
                        redge = redge
                else:
                    redge = peaks+np.amax(redge)  

                idx_redge, = np.where(redge>peaks[-1])

                redge[idx_redge] = peaks[-1]

            ledge = np.floor(ledge).astype(int)
            redge = np.ceil(redge).astype(int)

            for i in range(0, len(peaks)):
                
                idx=np.arange(ledge[i]-1, redge[i]+1) #Added and removed 1 just to consider eventual rounding
                x_inter = np.append(x_inter, idx)

                array_temp = np.append(replaced[j][ledge[i]-window-1:ledge[i]-1], \
                                       replaced[j][redge[i]+1:redge[i]+window+1])

                mean_temp = np.mean(array_temp)
                std_temp = np.std(array_temp)
                var_temp = np.var(array_temp)

                p_stat = np.abs(np.abs(mean_temp/var_temp)-1.)

                if p_stat <= 1e-2:
                    val = np.random.poisson(mean_temp, len(idx))
                    replaced[j][ledge[i]-1:redge[i]+1] = val
                else:
                    val = np.random.normal(mean_temp, std_temp, len(idx))
                    replaced[j][ledge[i]-1:redge[i]+1] = val

        return replaced

class filterdata:

    '''
    class for filter the detector TOD
    '''

    def __init__(self, data, cutoff, fs=None):
        
        '''
        See data_cleaned for parameters explanantion
        '''

        self.data = data
        self.cutoff = cutoff
        if fs is None:
            self.fs = 488.28125
        else:
            self.fs = fs
    
    def butter_filter(self, order, filter_type='highpass'):

        '''
        Highpass butterworth filter.
        order parameter is the order of the butterworth filter
        '''

        try:
            if filter_type.lower() not in ['highpass', 'lowpass']:
                raise InputError
        except InputError:
            print('The filter type choosen is not correct. Choose between highpass and lowpass')
            sys.exit(1)
        
        return sgn.butter(order, self.cutoff, btype=filter_type, analog=False, output='sos')

    def butter_filter_data(self, order=5, filter_type='highpass'):

        '''
        Data filtered with a butterworth filter 
        order parameter is the order of the butterworth filter
        '''
        sos = self.butter_filter(order, filter_type)
        
        filterdata = sgn.sosfiltfilt(sos, self.data)
        return filterdata

    def cosine_filter(self, f, filter_type='highpass'):

        '''
        Cosine filter
        '''

        if filter_type.lower() == 'highpass':
            if f < .5*self.cutoff:
                return 0
            elif 0.5*self.cutoff <= f  and f <= self.cutoff:
                return 0.5-0.5*np.cos(np.pi*(f-0.5*self.cutoff)*(self.cutoff-0.5*self.cutoff)**-1)
            elif f > self.cutoff:
                return 1

        elif filter_type.lower() == 'lowpass':
            if f < .5*self.cutoff:
                return 1
            elif 0.5*self.cutoff <= f  and f <= self.cutoff:
                return 0.5+0.5*np.cos(np.pi*(f-0.5*self.cutoff)/(0.5*self.cutoff))
            elif f > self.cutoff:
                return 0
    
    def cosine_filter_data(self, window=True, filter_type='highpass', \
                           window_type='hanning', beta_kaiser=0):

        '''
        Return an fft of the data using the cosine filter.
        Parameters:
        - window: if True the FFT is computed 
                  using one of the available windows
        - window_type: type of the window to be applied. Choose between
                       hanning, bartlett, blackman, hamming and kaiser
        - beta_kaiser: beta parameter for the kaiser window
        '''

        if window is True:
            if window_type.lower() == 'hanning':
                window_data = np.hanning(len(self.data[0]))
            elif window_type.lower() == 'bartlett':
                window_data = np.bartlett(len(self.data[0]))
            elif window_type.lower() == 'blackman':
                window_data = np.blackman(len(self.data[0]))
            elif window_type.lower() == 'hamming':
                window_data = np.hamming(len(self.data[0]))
            elif window_type.lower() == 'kaiser':
                window_data = np.kaiser(len(self.data[0]), beta_kaiser)

            fft_data = np.fft.rfft(self.data*window_data)
        else:
            fft_data = np.fft.rfft(self.data)

        fft_frequency = np.fft.rfftfreq(np.size(self.data[0]), 1/self.fs)

        vect = np.vectorize(self.cosine_filter, otypes=[np.float])

        ifft_data = np.fft.irfft(vect(fft_frequency, filter_type)*fft_data, len(self.data[0]))

        return ifft_data

    def filter_data(self, filter_type='highpass', filter_choice='cosine', \
                    cosine_window=True, cosine_filter_window='hanning', \
                    cosine_beta_kaiser=0, butter_order=5):

        '''
        Filter data with one the two filters available. 
        Options:
        - filter_type: choose if the filter should be an highpass or lowpass filter
        - filter_choice: choose between a cosine or a butterworth filter
        - cosine_filter_window: choose the filter window between
                                hanning, bartlett, blackman, hamming and kaiser
        - cosine_beta_kaiser: beta parameter for the kaiser window for a cosine window
        - butter_order: order for the butterworth filter
        '''

        try:
            if filter_type.lower() not in ['highpass', 'lowpass']:
                raise InputError
        except InputError:
            print('The filter type choosen is not correct. Choose between highpass and lowpass')
            sys.exit(1)

        try:
            if filter_choice.lower() not in ['cosine', 'butterworth']:
                raise InputError
        except InputError:
            print('The filter choice is not correct. Choose between cosine and butterworth')
            sys.exit(1)

        try:
            if cosine_window is True:
                window_list = ['hanning', 'bartlett', 'blackman', 'hamming', 'kaiser']
                if cosine_filter_window.lower() not in window_list:
                    raise InputError
        except InputError:
            print('The window choosen for the cosine filter is not correct. \
                   Choose between hanning, bartlett, blackman, hamming, kaiser')
            sys.exit(1)
        

        if filter_choice.lower() == 'cosine':
            
            filterdata = self.cosine_filter_data(cosine_window, filter_type, \
                                                 cosine_filter_window, cosine_beta_kaiser)
        
        elif filter_choice.lower() == 'butterworth':

            filterdata = self.butter_filter_data(butter_order, filter_type)

        return filterdata

class detector_trend():

    '''
    Class to detrend a TOD
    '''

    def __init__(self, data, remove_signal=False):
        
        self.data = data
        self.x = np.arange(len(self.data[0]))

        if isinstance(self.data, np.ma.core.MaskedArray):
            self.mask = self.data.mask

        if remove_signal:
            self.mask = np.zeros_like(data, dtype=bool)
            signal = despike(data, thres=None, hthres=None, width=250, rel_height=0.75)
            for i in range(np.shape(self.mask)[0]):
                ledge = np.floor(signal.param_peak[i]['left_ips']).astype(int)
                redge = np.floor(signal.param_peak[i]['right_ips']).astype(int)
                for j in range(len(ledge)):
                    self.mask[ledge[j]:redge[j]] = True
            self.mask_array = True
        else:
            self.mask_array = False

    def sine(self, x, a, b, c, d):

        return a*np.sin(b*x+c)+d

    def polyfit(self, y=None, order=6):

        '''
        Function to fit a trend line to a TOD
        '''
        
        if y is None:
            y = self.data

        if self.mask_array or isinstance(y, np.ma.core.MaskedArray):
            masked_array = np.ma.array(y, mask=self.mask)
            p = np.ma.polyfit(self.x, np.flip(masked_array.T, axis=1), order)
            p = np.flip(p)
        else:
            p = np.polynomial.polynomial.polyfit(self.x, np.flip(y.T, axis=1), order)

        y_fin = np.polynomial.polynomial.polyval(self.x, p)

        return y_fin, p

    def baseline(self, y=None, baseline_type='poly', interpolation_smooth=2.0,\
                 order=6, iter_val=100, tol=1e-3):

        import scipy.linalg as LA

        '''
        Routine to compute the baseline of the timestream. Based on the peakutils package for the polynomial
        Parameters:
        - baseline_type: choose the method for the baseline calculation. Either 'poly' for polynomial
                         or 'inter' for interpolation
        - interpolation_smooth: order of the interpolation. See scipy.interpolate.UnivariateSpliline 
                                for the meaning of this parameter
        - order: order of the polynominal
        - iter_val: number of iteration required to find the baseline
        - tol: tolerance to stop the iteration. The tolerance criteria is computed on the 
               coefficents of the polynomial
        '''

        if y is None:
            y = self.data
        
        if baseline_type.lower() == 'poly':
            
            coeffs = np.ones((int(order+1), np.shape(self.data)[0]))

            if isinstance(y, np.ma.core.MaskedArray):
                mask = y.mask

            for i in range(iter_val):

                base, coeffs_new = self.polyfit(y=y, order=order)

                if isinstance(y, np.ma.core.MaskedArray):
                    base = np.ma.array(base, mask=mask)

                if LA.norm(coeffs_new - coeffs) / LA.norm(coeffs) < tol:
                    coeffs = coeffs_new
                    break

                coeffs = coeffs_new
                y = np.minimum(y, base)

            self.offset = coeffs[-1,:]

            return np.polynomial.polynomial.polyval(self.x, coeffs)

        elif baseline_type.lower() == 'inter':

            points = np.ones(len(y))*np.mean(y)
            self.offset = np.zeros(np.shape(y)[0])
            base_inter = np.zeros_like(y)

            for j in range(np.shape(y)[0]):
                z = y[j]
                points = np.ones(len(z))*np.mean(z)
                for i in range(iter_val):
                    f = interpolate.UnivariateSpline(np.arange(len(z)), z, s=interpolation_smooth)
                    base = f(np.arange(len(z)))

                    if np.all(np.abs(base-points)<tol):
                        break

                    z = np.minimum(z, base)
                    points = base

                base_inter[j] = base
                self.offset[j] = np.mean(base[:1000])

            return base_inter

    
    def fit_residual(self, order=6, baseline=False, return_baseline=False, \
                     baseline_type='poly', interpolation_smooth=2.0,\
                     iter_val=100, tol=1e-3):

        '''
        Function to remove the trend polynomial from the TOD
        '''
        
        if baseline is True:
            baseline = self.baseline(baseline_type=baseline_type, interpolation_smooth=interpolation_smooth, \
                                     order=order, iter_val=iter_val, tol=tol)
        else:
            baseline = self.polyfit(order=order)[0]
        
        if return_baseline:
            return self.data-baseline, baseline
        else:
            return self.data-baseline

        def offset_level(self):

            '''
            Return the offset level of signal after running the baseline calculation
            '''

            return self.offset

class kidsutils:

    '''
    Class containing useful functions for KIDs
    '''

    def __init__(self, **kwargs):

        '''
        Possible arguments:
        - I: I of a detector as an array
        - Q: Q of a detector as an array
        - data_dir: directory with the detector data
        - roach_num: roach number of the detectors to be loaded
        - single: the channel parameter is intended as single channel or the upper value of 
                  all the channels to be loaded
        - chan: a single float number that can be interpreted as the channel to be loaded 
                or as the upper value of a list of channel to be loaded
        - first_sample: first sample of the TOD to be read
        - last_sample: last sample of the TOD to be read
        '''

        self.I = kwargs.get('I')
        self.Q = kwargs.get('Q')

        if self.I is None and self.Q is None:
            data_dir = kwargs.get('data_dir')
            roach_num = kwargs.get('roach_num')
            self.single = kwargs.get('single', False)
            self.chan = kwargs.get('chan')
            self.first_sample = kwargs.get('first_sample')
            self.last_sample = kwargs.get('last_sample')

            if (data_dir is not None and roach_num is not None and self.single is not None and \
                self.chan is not None and self.first_sample is not None and self.last_sample is not None):

                self.getTs(data_dir, roach_num, self.single, self.chan, self.first_sample, self.last_sample)

        self.Z = None 

        if self.I is not None and self.Q is not None:

            ### Complex Timestream ###
            self.Z = self.I + 1j*self.Q 
            
    def getTs(self, data_dir, roach_num, single, chan, start_samp, stop_samp):

        '''
        Load the detector timestreams using the following parameters:
        - data_dir: directory with the detector data
        - roach_num: roach number of the detectors to be loaded
        - single: the channel parameter is intended as single channel or the upper value of 
                  all the channels to be loaded
        - chan: a single float number that can be interpreted as the channel to be loaded 
                or as the upper value of a list of channel to be loaded
        - first_sample: first sample of the TOD to be read
        - last_sample: last sample of the TOD to be read

        If these parameters are given as input to the class, it is not necessary to call this method
        in the main script.
        '''

        fdir = gd.dirfile(data_dir)

        if single:
            self.I = np.zeros((1, int(stop_samp-start_samp)))
            self.Q = np.zeros((1, int(stop_samp-start_samp)))
            chan_number = np.array([chan])
        else:
            if isinstance(chan, float) or isinstance(chan, int):
                chan_number = np.arange(chan)
            else:
                chan_number = chan
            
            self.I = np.zeros((len(chan_number), int(stop_samp-start_samp)))
            self.Q = np.zeros((len(chan_number), int(stop_samp-start_samp)))

        for i in range(len(chan_number)):
            self.I[i] = fdir.getdata("i_kid"+"%04d" % (chan_number[i],)+"_roach"+str(roach_num), \
                                     first_sample = start_samp,num_samples=int(stop_samp-start_samp))
            self.Q[i] = fdir.getdata("q_kid"+"%04d" % (chan_number[i],)+"_roach"+str(roach_num), \
                                     first_sample = start_samp,num_samples=int(stop_samp-start_samp))

    def rotatePhase(self):

        '''
        Rotate phase for a KID
        '''

        X = self.I+1j*self.Q
        phi_avg = np.arctan2(np.mean(self.Q),np.mean(self.I))
        E = X*np.exp(-1j*phi_avg)
        self.I_rot = E.real
        self.Q_rot = E.imag

    def KIDphase(self):

        '''
        Compute the phase of a KID. This is proportional to power, in particular
        Power = Phase/Responsivity
        '''

        phibar = np.arctan2(np.mean(self.Q),np.mean(self.I))
        self.rotatePhase()
        phi = np.arctan2(self.Q_rot, self.I_rot)

        return phi-phibar

    def KIDmag(self):

        ''' 
        Compute the magnitude response of a KID
        '''

        return np.sqrt(self.I**2+self.Q**2)

    def loadBinarySweepData(self, path_to_sweep, vna = True):
        '''
        Function to load the the sweep from a sweep.
        Parameters:
        - path_to_sweep: path with the sweep file
        - vna: if True, the VNA sweep data is used.

        From plotPipeline.py - Sam Gordon
        '''
        try:

            sweep_freqs = np.loadtxt(os.path.join(path_to_sweep, "sweep_freqs.dat"), dtype = "float")
            if vna:
                dac_freqs = np.loadtxt(os.path.join(path_to_sweep, "vna_freqs.dat"), dtype = "float")
            else:
                dac_freqs = np.loadtxt(os.path.join(path_to_sweep, "bb_targ_freqs.dat"), dtype = "float")
            chan_I = np.zeros((len(sweep_freqs),len(dac_freqs)))
            chan_Q = np.zeros((len(sweep_freqs),len(dac_freqs)))

            for i in range(len(sweep_freqs)):
                name = str(int(sweep_freqs[i])) +'.dat'
                file_path = os.path.join(path_to_sweep, name)
                raw = open(file_path, 'rb')
                data = np.fromfile(raw,dtype = '<f')
                chan_I[i] = data[::2]
                chan_Q[i] = data[1::2]
        except ValueError:
            return chan_I, chan_Q
        return chan_I, chan_Q

    def get_df(self, **kwargs):

        '''
        Compute df of a timestream and return frequency and dissipation direction
        Parameters:
        - s21_f: the complex sweep stream
        - path_to_sweep: if there is no s21_f it is possible to give the path for sweep files
                         to estimate s21_f
        - vna: if True, the sweep files to be loaded come from a VNA sweep
        - despike: if True, the newly loaded s21_f are despiked using a butterworth filter 
                   with the following arguments:
                   - std: standard deviation in unit of sigma to exclude outliers. Default at 5
                   - cutoff: cutoff frequency of the filter. Default at  0.1Hz
        - delta_f: size in Hz of the df. Default at 1000
        - idx_shift: index for shifting. Default at 2
        - timestream: detector complex timestream. If not set, it will be used self.Z 
        - window: number of samples to exclude at the edges to look for the resonance in the sweep. 
                  If a float, the number of samples to exclude is applied symmetrically at both edges.
                  Else, the first element is the left edge and the second is the right. The window is the
                  same for each resonance. Default is a float at 0
        '''

        path_to_sweep = kwargs.get('path_to_sweep')
        vna = kwargs.get('vna', False)

        if path_to_sweep is None:
            s21_f = kwargs.get('s21_f')
            s21_f_real, s21_f_imag = s21_f.real, s21_f.imag
        else:
            s21_f_real_temp, s21_f_imag_temp = self.loadBinarySweepData(path_to_sweep, vna)

            filtering = kwargs.get('filtering', True)

            if filtering:
                cutoff = kwargs.get('cutoff', 0.1)
                
                s21_f_real = np.zeros_like(s21_f_real_temp.T)
                s21_f_imag = np.zeros_like(s21_f_imag_temp.T)

                for i in range(len(s21_f_real)):
                    s21_real_filt = filterdata(s21_f_real_temp.T[i], cutoff)
                    s21_imag_filt = filterdata(s21_f_imag_temp.T[i], cutoff)

                    s21_f_real[i] = s21_real_filt.butter_filter_data(order=4, filter_type="lowpass") 
                    s21_f_imag[i] = s21_imag_filt.butter_filter_data(order=4, filter_type="lowpass")

                del s21_f_real_temp
                del s21_f_imag_temp
            else:

                s21_f_real = s21_f_real_temp.T
                s21_f_imag = s21_f_imag_temp.T


        try: 
            single_chan = self.single
        except AttributeError:
            single_chan = kwargs.get('single', True)

        try: 
            channel = self.chan
        except AttributeError:
            channel = kwargs.get('chan')

        delta_f = kwargs.get('df', 1000)
        shift_idx = kwargs.get('shift_idx', 2)

        window = kwargs.get('window', 0.)

        if isinstance(window, float) or isinstance(window, int):
            window = np.rint(window)
            window = np.array([window, window], dtype=int)
        else:
            if isinstance(window, np.ndarray) is False:
                window = np.array(window, dtype=int)
            else:
                window = window

        timestream = kwargs.get('timestream')

        if timestream is None:
            timestream = self.Z
        
        if single_chan:
            dI, dQ = np.array([np.diff(s21_f_real[channel])]), np.array([np.diff(s21_f_imag[channel])])
            Mag = np.array([np.sqrt(s21_f_real[channel]**2+s21_f_imag[channel]**2)])
        else:
            if isinstance(channel, int) or isinstance(channel, float):
                channel = int(channel)
            else:
                channel = channel

            dI, dQ = np.diff(s21_f_real[channel]), np.diff(s21_f_imag[channel])
            Mag = np.sqrt(s21_f_real[channel]**2+s21_f_imag[channel]**2)

        dIdf, dQdf = dI/delta_f, dQ/delta_f        
        dMag = np.sqrt(dIdf**2+dQdf**2)
        

        dIdf, dQdf, dMag = np.roll(dIdf,-shift_idx, axis=1), np.roll(dQdf,-shift_idx, axis=1), \
                           np.roll(dMag,-shift_idx, axis=1)

        min_indices = np.zeros(np.shape(timestream)[0]).astype("int")
        dIdf_min, dQdf_min, dMag_min = np.zeros(len(dIdf)), np.zeros(len(dIdf)), np.zeros(len(dIdf))

        for i in range(np.shape(timestream)[0]):
            min_idx = np.where(Mag[i][window[0]:-window[1]]==min(Mag[i][window[0]:-window[1]]))[0][0]
            min_indices[i] = min_idx
            dIdf_min[i] = dIdf[i][window[0]+min_indices[i]]
            dQdf_min[i] = dQdf[i][window[0]+min_indices[i]]
            dMag_min[i] = dMag[i][window[0]+min_indices[i]]
        
        df_x = np.copy((timestream.real.T*dIdf_min + timestream.imag.T*dQdf_min)/dMag_min**2).T
        df_y  = np.copy((timestream.imag.T*dIdf_min - timestream.real.T*dQdf_min)/dMag_min**2).T

        del dI
        del dQ
        del Mag
        
        return df_x, df_y

    def despike_sweeps(self, I_targ, Q_targ, std=5, cutoff=0.1):

        '''
        Despike the sweep using a butterworth filter
        Parameters:
        - I_targ: real component of the the sweep
        - Q_targ: imaginary component of the the sweep
        - std: standard deviation in unit of sigma
        - cutoff: cutoff frequency of the filter
        '''

        I_targ_ds,Q_targ_ds = np.array([]), np.array([]) 
        for i in range(len(I_targ)):
            I_temp, Q_temp = I_targ[i], Q_targ[i]
            I_idx = self.removeStd(I_temp,s=std)
            I_temp, Q_temp = I_temp[I_idx],Q_temp[I_idx]
            Q_idx = self.removeStd(Q_temp,s=std)
            I_temp, Q_temp = I_temp[Q_idx],Q_temp[Q_idx]

            I_filt = filterdata(I_temp, cutoff)
            Q_filt = filterdata(Q_temp, cutoff)

            I_ds_f = I_filt.butter_filter_data(order=4, filter_type="highpass") 
            Q_ds_f = Q_filt.butter_filter_data(order=4, filter_type="highpass")

            I_targ_ds = np.append(I_targ_ds, I_ds_f)
            Q_targ_ds = np.append(Q_targ_ds, Q_ds_f)

        return I_targ_ds, Q_targ_ds

    def removeStd(self, var, s=5.):
        std = np.std(var)
        m = np.mean(var)
        return np.where(np.abs(var-m)<std*s)

