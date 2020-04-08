import numpy as np
import scipy.signal as sgn
import sys

class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class InputError(Error):
    """Exception raised for errors in the input. """
    pass

class data_cleaned():

    '''
    Class to clean the detector TOD using the functions in 
    the next classes. Check them for more explanations
    '''

    def __init__(self, data, fs, cutoff, detlist, polynomialorder, despike, sigma, prominence):

        self.data = data                #detector TOD
        self.fs = float(fs)             #frequency sampling of the detector
        self.cutoff = float(cutoff)     #cutoff frequency of the highpass filter
        self.detlist = detlist          #detector name list
        self.polynomialorder = polynomialorder #polynomial order for fitting
        self.sigma = sigma                  #height in std value to look for spikes
        self.prominence = prominence        #prominence in std value to look for spikes
        self.despike = despike              #if True despikes the data 
        

    def data_clean(self):

        '''
        Function to return the cleaned TOD as numpy array
        '''
        cleaned_data = np.zeros_like(self.data)


        if np.size(self.detlist) == 1:
            det_data = detector_trend(self.data)
            if self.polynomialorder != 0:
                residual_data = det_data.fit_residual(order=self.polynomialorder)
            else:
                residual_data = self.data

            if self.despike is True:
                desp = despike(residual_data)
                data_despiked = desp.replace_peak(hthres=self.sigma, pthres=self.prominence)
            else:
                data_despiked = residual_data.copy()

            if self.cutoff != 0:
                filterdat = filterdata(data_despiked, self.cutoff, self.fs)
                cleaned_data = filterdat.ifft_filter(window=True)
            else:
                cleaned_data = data_despiked

            return cleaned_data

        else:
            for i in range(np.size(self.detlist)):
                det_data = detector_trend(self.data[i,:])
                residual_data = det_data.fit_residual()

                desp = despike(residual_data)
                data_despiked = desp.replace_peak()

                filterdat = filterdata(data_despiked, self.cutoff, self.fs)
                cleaned_data[i,:] = filterdat.ifft_filter(window=True)

            return cleaned_data
        
class despike():

    '''
    Class to despike the TOD
    '''

    def __init__(self, data):

        self.data = data

    def findpeak(self, thres=5, hthres=5, pthres=None, width=np.array([1, 10])):

        '''
        This function finds the peak in the TOD. The optional arguments are the standard 
        from scipy.signal.find_peaks

        hthresh, pthres and thresh are measured in how many std the height, the prominence 
        or the threshold (distance from its neighbouring samples) of the peak is computed. 
        The height of the peak is computed with respect to the mean of the signal        
        '''

        val = (self.data-np.mean(self.data))/np.mean(self.data)

        y_std = np.std(self.data[val<0.3])
        y_mean = np.mean(self.data[val<0.3])
        
        data_to_despike = self.data-y_mean

        if hthres is not None:
            hthres = hthres*y_std
        if pthres is not None:
            pthres = pthres*y_std
        if thres is not None:
            thresh = thres*y_std
        if width is not None:
            width = width

        index, param = sgn.find_peaks(np.abs(data_to_despike), height = hthres, prominence = pthres, \
                                      threshold = thresh, width=width)

        return index

    def peak_width(self, peaks, thres=5, hthres=5, pthres=0, window = 100):


        '''
        Function to estimate the width of the peaks.
        Window is the parameter used by the algorith to find the minimum 
        left and right of the peak. The minimum at left and right is used
        to compute the width of the peak
        '''

        y_mean = np.mean(self.data)
        if np.amin(self.data) > 0:
            data_to_despike = self.data-y_mean
        else:
            data_to_despike = self.data.copy()
        param = sgn.peak_widths(np.abs(data_to_despike),peaks, rel_height = 1.0)

        ledge = np.array([], dtype='int')
        redge = np.array([], dtype='int')

        for i in range(len(peaks)):
            try:
                left_edge, = np.where(np.abs(data_to_despike[peaks[i]-window:peaks[i]]) == \
                                      np.amin(np.abs(data_to_despike[peaks[i]-window:peaks[i]])))
                right_edge, = np.where(np.abs(data_to_despike[peaks[i]:peaks[i]+window]) == \
                                       np.amin(np.abs(data_to_despike[peaks[i]:peaks[i]+window])))
            except ValueError:
                left_edge = 0
                right_edge = 0

            left_edge += (peaks[i]-window)
            right_edge += peaks[i]

            ledge = np.append(ledge, left_edge[-1])
            redge = np.append(redge, right_edge[-1])

        return param[0].copy(), ledge, redge

    def replace_peak(self, thres=5, hthres=5, pthres=None, peaks = np.array([]), widths = np.array([0, 10])):

        '''
        This function replaces the spikes data with noise realization. Noise can be gaussian
        or poissonian based on the statistic of the data
        - widths may be a 1D array with 2 values (the minimum expect width and the maximum expected width) or 
          None. In this case the width will be computed automatically by the code
        '''

        x_inter = np.array([], dtype = 'int')

        ledge = np.array([], 'int')
        redge = np.array([], 'int')
        replaced = self.data.copy()

        if np.size(peaks) == 0:
            peaks = self.findpeak(thres=thres, hthres=hthres, pthres=pthres)
        
        if isinstance(widths, np.ndarray) is False:
            widths, ledge, redge = self.peak_width(peaks=peaks, thres=thres, hthres=hthres, pthres=pthres)
        else:
            ledge = np.ones_like(peaks)*peaks-np.amax(widths)

            if peaks[-1]+np.amax(widths) < len(self.data):
                redge = np.ones_like(peaks)*peaks+np.amax(widths)
            else:
                for j in range(len(peaks)):
                    if peaks[j]+np.amax(widths) > len(self.data):
                        redge = len(self.data)-1
                    else:
                        redge = peaks[j]+np.amax(widths)

            ledge = np.floor(ledge).astype(int)
            redge = np.floor(redge).astype(int)

        for i in range(0, len(peaks)):

            x_inter = np.append(x_inter, np.arange(ledge[i], redge[i]))
            replaced[ledge[i]:redge[i]] = (replaced[ledge[i]]+\
                                           replaced[redge[i]])/2.

        val = (replaced-np.mean(replaced))/np.mean(replaced)

        final_mean = np.mean(replaced[val<0.3])
        final_std = np.std(replaced[val<0.3])
        final_var = np.var(replaced[val<0.3])

        p_stat = np.abs(final_mean/final_var-1.)

        if p_stat <=1e-2:
            '''
            This means that the variance and the mean are approximately the 
            same, so the distribution is Poissonian.
            '''
            mu = (final_mean+final_var)/2.
            y_sub = np.random.poisson(mu, len(x_inter))
        else:
            y_sub = np.random.normal(final_mean, final_std, len(x_inter))

        if np.size(y_sub) > 0:
            replaced[x_inter] = y_sub

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

        vect = np.vectorize(self.cosine_filter)

        ifft_data = np.fft.irfft(vect(fft_frequency)*fft_data, len(self.data))

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

    def __init__(self, data):

        self.data = data
        self.x = np.arange(len(self.data))

    def polyfit(self, y=None, order=6):

        '''
        Function to fit a trend line to a TOD
        '''
        
        if y is None:
            y = self.data

        p = np.polyfit(self.x, y, order)
        poly = np.poly1d(p)
        y_fin = poly(self.x)

        return y_fin, p

    def baseline(self, y=None, order=6, iter_val=100, tol=1e-3):

        import scipy.linalg as LA

        '''
        Routine to compute the baseline of the timestream. Based on the peakutils package
        Parameters:
        - order: order of the polynominal
        - iter_val: number of iteration required to find the baseline
        - tol: tolerance to stop the iteration. The tolerance criteria is computed on the 
               coefficents of the polynomial
        '''

        coeffs = np.ones(int(order+1))

        if y is None:
            y = self.data
        
        for i in range(iter_val):

            base, coeffs_new = self.polyfit(y=y, order=order)

            if LA.norm(coeffs_new - coeffs) / LA.norm(coeffs) < tol:
                coeffs = coeffs_new
                break

            coeffs = coeffs_new
            y = np.minimum(y, base)

        poly = np.poly1d(coeffs)

        return poly(self.x)
    
    def fit_residual(self, order=6, baseline=False):

        '''
        Function to remove the trend polynomial from the TOD
        '''

        if baseline is True:
            baseline = self.baseline(order=order)
        else:
            baseline = self.polyfit(order=order)[0]

        return self.data-baseline

class kidsutils():

    '''
    Class containing useful functions for KIDs
    '''

    def rotatePhase(self, I, Q):

        '''
        Rotate phase for a KID
        '''

        X = I+1j*Q
        phi_avg = np.arctan2(np.mean(Q),np.mean(I))
        E = X*np.exp(-1j*phi_avg)
        I = E.real
        Q = E.imag

        return I, Q

    def KIDphase(self, I, Q):

        '''
        Compute the phase of a KID. This is proportional to power, in particular
        Power = Phase/Responsivity
        '''

        phibar = np.arctan2(np.mean(Q),np.mean(I))
        I_rot, Q_rot = self.rotatePhase(I, Q)
        phi = np.arctan2(Q,I)

        return phi-phibar

    def KIDmag(self, I, Q):

        ''' 
        Compute the magnitude response of a KID
        '''

        return np.sqrt(I**2+Q**2 )
