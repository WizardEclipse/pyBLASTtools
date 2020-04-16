import numpy as np
import scipy.signal as sgn
import sys
from scipy import interpolate

class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class InputError(Error):
    """Exception raised for errors in the input. """
    pass
        
class despike():

    '''
    Class to despike the TOD
    '''

    def __init__(self, data, remove_mean=True, mean_std_distance=0.3, hthres=4, thres=4, pthres=None, \
                 width=np.array([1,10]), full_width=False, rel_height=0.5):

        '''
        These are the parameters for this class:

        - data: signal that needs to be analyzed
        - remove_mean: a parameter to choose if the mean is removed from the signal that needs 
                       to be analyzed
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
        self.mean_std_distance = mean_std_distance

        if self.mean_std_distance < 1e-5 or self.mean_std_distance > 1:
            self.mean_std_distance = 0.3

        val = (self.data-np.mean(self.data))/np.mean(self.data)

        self.signal_std = np.std(self.data[val<mean_std_distance])
        self.signal_mean = np.mean(self.data[val<mean_std_distance])
        
        if remove_mean:
            self.signal = self.data-self.signal_mean
        else:
            self.signal = self.data.copy()

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

        self.idx_peak, self.param_peak = sgn.find_peaks(np.abs(self.signal), height=self.hthres, \
                                                        prominence=self.pthres, \
                                                        threshold=self.thres, width=self.width, \
                                                        rel_height=rel_height)

        if full_width:
            param_temp = sgn.peak_widths(np.abs(self.signal), peaks=self.idx_peak, rel_height=1.)

            ledge = np.append(param_temp[-2][0], np.maximum(param_temp[-2][1:], param_temp[-1][:-1]))
            redge = np.append(np.minimum(param_temp[-2][1:], param_temp[-1][:-1]), param_temp[-1][-1])

            self.param_peak['widths'] = param_temp[0]
            self.param_peak['left_ips'] = ledge
            self.param_peak['right_ips'] = redge

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

        replaced = self.signal.copy()

        if peaks is None:
            peaks = self.idx_peak

        if ledge is None:
            ledge = self.param_peak['left_ips']
        else:
            if isinstance(ledge, np.ndarray):
                if len(ledge) != len(peaks):
                    ledge = peaks-np.amax(ledge)
                else:
                    ledge = ledge
            else:
                ledge = peaks-np.amax(ledge)            

        if redge is None:
            redge = self.param_peak['right_ips']
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

            array_temp = np.append(replaced[ledge[i]-window-1:ledge[i]-1], \
                                   replaced[redge[i]+1:redge[i]+window+1])

            mean_temp = np.mean(array_temp)
            std_temp = np.std(array_temp)
            var_temp = np.var(array_temp)

            p_stat = np.abs(np.abs(mean_temp/var_temp)-1.)

            if p_stat <= 1e-2:
                val = np.random.poisson(mean_temp, len(idx))
                replaced[ledge[i]-1:redge[i]+1] = val
            else:
                val = np.random.normal(mean_temp, std_temp, len(idx))
                replaced[ledge[i]-1:redge[i]+1] = val

        return replaced

class filterdata():

    '''
    class for filter the detector TOD
    '''

    def __init__(self, data, cutoff, fs):
        
        '''
        See data_cleaned for parameters explanantion
        '''

        self.data = data
        self.cutoff = cutoff
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
        
        nyq = 0.5*self.fs
        normal_cutoff = self.cutoff / nyq
        b, a = sgn.butter(order, normal_cutoff, btype=filter_type, analog=False)
        return b, a

    def butter_filter_data(self, order=5, filter_type='highpass'):

        '''
        Data filtered with a butterworth filter 
        order parameter is the order of the butterworth filter
        '''
        b, a = self.butter_filter(order, filter_type)
        filterdata = sgn.lfilter(b, a, self.data)
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
                window_data = np.hanning(len(self.data))
            elif window_type.lower() == 'bartlett':
                window_data = np.bartlett(len(self.data))
            elif window_type.lower() == 'blackman':
                window_data = np.blackman(len(self.data))
            elif window_type.lower() == 'hamming':
                window_data = np.hamming(len(self.data))
            elif window_type.lower() == 'kaiser':
                window_data = np.kaiser(len(self.data), beta_kaiser)

            fft_data = np.fft.rfft(self.data*window_data)
        else:
            fft_data = np.fft.rfft(self.data)

        fft_frequency = np.fft.rfftfreq(np.size(self.data), 1/self.fs)

        vect = np.vectorize(self.cosine_filter, otypes=[np.float])

        ifft_data = np.fft.irfft(vect(fft_frequency, filter_type)*fft_data, len(self.data))

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
        self.x = np.arange(len(self.data))

        if remove_signal:
            signal = despike(data, thres=None, hthres=None, width=250, rel_height=0.75)
            self.mask = np.zeros_like(data, dtype=bool)
            ledge = np.floor(signal.param_peak['left_ips']).astype(int)
            redge = np.floor(signal.param_peak['right_ips']).astype(int)
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

        if self.mask_array:
            masked_array = np.ma.array(y, mask=self.mask)
            p = np.ma.polyfit(self.x, masked_array, order)
        else:
            p = np.polyfit(self.x, y, order)
        
        poly = np.poly1d(p)
        y_fin = poly(self.x)

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
            
            coeffs = np.ones(int(order+1))

            for i in range(iter_val):

                base, coeffs_new = self.polyfit(y=y, order=order)

                if LA.norm(coeffs_new - coeffs) / LA.norm(coeffs) < tol:
                    coeffs = coeffs_new
                    break

                coeffs = coeffs_new
                y = np.minimum(y, base)

            poly = np.poly1d(coeffs)

            self.offset = coeffs[-1]

            return poly(self.x)

        elif baseline_type.lower() == 'inter':

            points = np.ones(len(y))*np.mean(y)

            for i in range(iter_val):
                f = interpolate.UnivariateSpline(np.arange(len(y)), y, s=interpolation_smooth)
                base = f(np.arange(len(y)))

                if np.all(np.abs(base-points)<tol):
                    break

                y = np.minimum(y, base)
                points = base

            self.offset = np.mean(base[:1000])

            return base
    
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

class kidsutils():

    '''
    Class containing useful functions for KIDs
    '''

    def __init__(self, I, Q):

        self.I = I
        self.Q = Q

        self.phase = self.KIDphase()
        self.mag = self.KIDmag()

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