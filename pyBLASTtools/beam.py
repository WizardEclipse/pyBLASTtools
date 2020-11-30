import numpy as np
from scipy.linalg import svd
from scipy.optimize import least_squares
from photutils import find_peaks
from astropy.stats import sigma_clipped_stats
from lmfit import Parameters, minimize

def centroid(map_data, pixel1_coord, pixel2_coord, threshold=0.275):

    '''
    For more information about centroid calculation see Shariff, PhD Thesis, 2016
    '''

    data = map_data.copy()

    data[np.isnan(data)] = 0

    maxval = np.max(data)

    y_max, x_max = np.where(data == maxval)

    gt_inds = np.where(data > threshold*maxval)

    weight = np.zeros((data.shape[0], data.shape[1]))
    weight[gt_inds] = 1.
    a = data[gt_inds]
    flux = np.sum(a)

    x_coord_max = np.rint(np.amax(pixel1_coord))+1
    x_coord_min = np.rint(np.amin(pixel1_coord))

    x_arr = np.arange(x_coord_min, x_coord_max)

    y_coord_max = np.rint(np.amax(pixel2_coord))+1
    y_coord_min = np.rint(np.amin(pixel2_coord))

    y_arr = np.arange(y_coord_min, y_coord_max)

    xx, yy = np.meshgrid(x_arr, y_arr)
    
    x_c = np.sum(xx*weight*data)/flux
    y_c = np.sum(yy*weight*data)/flux

    return x_c, y_c


class beam(object):

    def __init__(self, data, param = None):

        '''
        Class to handle beam operations on a map
        - data: Map to be analyzed
        - param: parameter for the gaussian model of a beam
        '''

        self.data = data
        self.param = param

        self.xgrid = np.arange(len(self.data[0,:]))
        self.ygrid = np.arange(len(self.data[:,0]))
        self.xy_mesh = np.meshgrid(self.xgrid,self.ygrid)

    def multivariate_gaussian_2d(self, params):

        (x, y) = self.xy_mesh
        for i in range(int(np.size(params)/6)):
            j = i*6
            amp = params[j]
            xo = float(params[j+1])
            yo = float(params[j+2])
            sigma_x = params[j+3]
            sigma_y = params[j+4]
            theta = params[j+5]   
            a = (np.cos(theta)**2)/(2*sigma_x**2)+(np.sin(theta)**2)/(2*sigma_y**2)
            b = -(np.sin(2*theta))/(4*sigma_x**2)+(np.sin(2*theta))/(4*sigma_y**2)
            c = (np.sin(theta)**2)/(2*sigma_x**2)+(np.cos(theta)**2)/(2*sigma_y**2)
            if i == 0:
                multivariate_gaussian = amp*np.exp(-(a*((x-xo)**2)+2*b*(x-xo)*(y-yo)+c*((y-yo)**2)))
            else:
                multivariate_gaussian += amp*np.exp(-(a*((x-xo)**2)+2*b*(x-xo)*(y-yo)+c*((y-yo)**2)))
        
        return np.ravel(multivariate_gaussian)

    def residuals(self, params, x, y, err, maxv):
        dat = self.multivariate_gaussian_2d(params)
        index, = np.where(y>=0.2*maxv)
        return (dat[index]-y[index])**2 / err[index]

    def peak_finder(self, map_data, mask_pf = None):

        '''
        Function to generate initial parameters of gaussian(s) to be used as guesses in the fitting algorithm.
        The code is searching automatically for peaks location using astropy and photutils routines. 
        Parameters:
        - map_data: map to be searched for gaussian(s)
        - mask_pf: a mask to be used in the searching algorithm. The alorithm is looking for gaussians only where 
                   the mask is true. In case of an iterative fitting, the mask is updated everytime to mask out 
                   the gaussians that have been already found.  
        '''

        x_lim = np.size(self.xgrid)
        y_lim = np.size(self.ygrid)
        fact = 20.

        bs = np.array([int(np.floor(y_lim/fact)), int(np.floor(x_lim/fact))])

        mean, median, std = sigma_clipped_stats(self.data, sigma=3.0)
        threshold = median+(5.*std)
        if mask_pf is None:
            tbl = find_peaks(map_data, threshold, box_size=bs)
            mask_pf = np.zeros_like(self.xy_mesh[0])
        else:
            self.mask = mask_pf.copy()
            tbl = find_peaks(map_data, threshold, box_size=bs, mask = self.mask)
        tbl['peak_value'].info.format = '%.8g'

        guess = np.array([])

        x_i = np.array([])
        y_i = np.array([])

        for i in range(len(tbl['peak_value'])):
            guess_temp = np.array([tbl['peak_value'][i], self.xgrid[tbl['x_peak'][i]], \
                                  self.ygrid[tbl['y_peak'][i]], 1., 1., 0.])
            guess = np.append(guess, guess_temp)
            index_x = self.xgrid[tbl['x_peak'][i]]
            index_y = self.ygrid[tbl['y_peak'][i]]
            x_i = np.append(x_i, index_x)
            y_i = np.append(y_i, index_y)
            mask_pf[index_y-bs[1]:index_y+bs[1], index_x-bs[0]:index_x+bs[0]] = True

            if self.param is None:
                self.param = guess_temp
                self.mask = mask_pf.copy()

            else:
                self.param = np.append(self.param, guess_temp)
                self.mask = np.logical_or(self.mask, mask_pf)


    def residual_lmfit(self, p):

        params = np.array([])

        v = p.valuesdict()

        for k in v.keys():
            params = np.append(params, v[k])

        dat = self.multivariate_gaussian_2d(params)
        y = np.ravel(self.data.copy())
        return dat-y 


    def test_lmfit(self, guess, d=None, method='cg'):

        fit_params = Parameters()
        fit_params.add('amp', value=guess[0], max=guess[0]*1.2, min=guess[0]*0.8)
        fit_params.add('xc', value=float(guess[1]), max=guess[1]+20, min=guess[1]-20)
        fit_params.add('yc', value=float(guess[2]), max=guess[2]+20, min=guess[2]-20)
        fit_params.add('sigma_x', value=guess[3], max=20.00, min=1.00)
        fit_params.add('sigma_y', value=guess[4], max=20.00, min=1.00)
        fit_params.add('theta', value=guess[5], max=2*np.pi, min=0.00)

        if d:
            values = np.ravel(self.data.copy())
        else:
            values = np.ravel(self.data.copy())
            values[values == 0] = np.nan

        out = minimize(self.residual_lmfit, fit_params, method=method, nan_policy='omit')

        return out

    def create_bounds(self, min_amplitude=0.2, max_amplitude=1.2, window_pixel=20, min_sigma=1, max_sigma=20):

        '''
        Create bounds for the parameter of the gaussian fitting
        - min_amplitude: minimum amplitude of the gaussian. The bound is created taking the initial guess and scale 
                         by this factor
        - max_amplitude: maximum amplitude of the gaussian. The bound is created taking the initial guess and scale 
                         by this factor
        - window_pixel: the number of pixels, left and right, up and down, to look for the the center of the gaussian
        - min_sigma: the minimum size of the sigma in pixel size (valid for both sigma_x and sigma_y)
        - max_sigma: the maximum size of the sigma in pixel size (valid for both sigma_x and sigma_y)
        '''

        lower_bounds = []
        upper_bounds = []

        for i in range(int(len(self.param)/6)):
            start_idx = i*6
            lower_bounds.append(min_amplitude*self.param[start_idx+i])
            lower_bounds.append(self.param[start_idx+i+1]-window_pixel)
            lower_bounds.append(self.param[start_idx+i+2]-window_pixel)
            lower_bounds.append(min_sigma)
            lower_bounds.append(min_sigma)
            lower_bounds.append(0.)
            upper_bounds.append(max_amplitude*self.param[start_idx+i])
            upper_bounds.append(self.param[start_idx+i+1]+window_pixel)
            upper_bounds.append(self.param[start_idx+i+2]+window_pixel)
            upper_bounds.append(max_sigma)
            upper_bounds.append(max_sigma)
            upper_bounds.append(2*np.pi)

        self.bounds = (lower_bounds, upper_bounds)
    
    def fit(self, bounds=False, min_amplitude=0.2, max_amplitude=1.2, window_pixel=20, min_sigma=1, max_sigma=20):

        '''
        Base function to fit a 2D image.
        If the method does not converge an error message is returned 
        '''

        try:
            if bounds:
                self.create_bounds(min_amplitude, max_amplitude, window_pixel, min_sigma, max_sigma)
                p = least_squares(self.residuals, x0=self.param, \
                                  args=(self.xy_mesh, np.ravel(self.data),\
                                        np.ones(len(np.ravel(self.data))), np.amax(self.data)), \
                                  bounds=self.bounds, method='trf')
            else:
                p = least_squares(self.residuals, x0=self.param, \
                                  args=(self.xy_mesh, np.ravel(self.data),\
                                        np.ones(len(np.ravel(self.data))), np.amax(self.data)), \
                                  method='lm')
            _, s, VT = svd(p.jac, full_matrices=False)
            threshold = np.finfo(float).eps * max(p.jac.shape) * s[0]
            s = s[s > threshold]
            VT = VT[:s.size]
            var = np.dot(VT.T / s**2, VT)
            
            return p, var
        except np.linalg.LinAlgError:
            msg = 'Fit not converged'
            return msg, 0
        except ValueError:
            msg = 'Too Many parameters'
            return msg, 0

    def beam_fit(self, recursive=False, mask_pf= None, iter_num=5, \
                 bounds=False, min_amplitude=0.2, max_amplitude=1.2, window_pixel=20, min_sigma=1, max_sigma=20):

        '''
        Method to fit a beam given a map. 
        Possible input parameters:
        - recursive: If True, the code is iteratively searching for peaks after each fit run.
                     until the number of peaks is equal to zero or the number of iteration is 
                     larger than the iter_num parameter
        - mask_pf: a mask to hide some pixel in the algorithm for finding the peaks and so the gaussian.
                   If None, no mask is applied in the first step of a recursive search
        - iter_num: The max number of iteration to search for gaussian components       
        '''

        if self.param is not None:
            peak_found = np.size(self.param)/6
            force_fit = True
        else:
            self.peak_finder(map_data = self.data, mask_pf = mask_pf)
            peak_number_ini = np.size(self.param)/6
            peak_found = peak_number_ini
            force_fit = False

        if recursive:
            iter_count = 0
            while peak_found > 0:
                fit_param, var = self.fit(bounds, min_amplitude, max_amplitude, window_pixel, min_sigma, max_sigma)
                if isinstance(fit_param, str):
                    msg = 'fit not converged'
                    break
                else:
                    fit_data = self.multivariate_gaussian_2d(fit_param.x).reshape(np.outer(self.ygrid, self.xgrid).shape)

                    if force_fit is False:
                        res = self.data-fit_data

                        self.peak_finder(map_data=res)

                        peak_number = np.size(self.param)/6
                        peak_found = peak_number-peak_number_ini
                        peak_number_ini = peak_number
                    else:
                        peak_found = -1

                if iter_count > iter_num:
                    break
                
                iter_count += 1
                
        else:
            fit_param, var = self.fit(bounds, min_amplitude, max_amplitude, window_pixel, min_sigma, max_sigma)
            if isinstance(fit_param, str):
                msg = 'fit not converged'
            else:
                fit_data = self.multivariate_gaussian_2d(fit_param.x).reshape(np.outer(self.ygrid, self.xgrid).shape)

        if isinstance(fit_param, str):
            return msg, 0, 0
        else:
            return fit_data, fit_param, var
        






