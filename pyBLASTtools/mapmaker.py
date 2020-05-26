import numpy as np
import sys
from astropy import wcs
from astropy.convolution import Gaussian2DKernel, convolve

class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class MapError(Error):
    """Exception raised for errors in the input. """
    pass

class wcs_world():

    '''
    Class to generate a wcs using astropy routines.
    '''

    def __init__(self, **kwargs):

        # Build a WCS system 

        self.w = kwargs.get('wcs')
        
        if self.w is None:
            self.w = wcs.WCS(naxis=2)
            self.w.wcs.crpix = kwargs.get('crpix')
            self.w.wcs.cdelt = kwargs.get('cdelt')
            self.w.wcs.crval = kwargs.get('crval', np.zeros(2))

            telcoord = kwargs.get('telcoord', False)
            
            if telcoord is False:
                self.w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
            else:
                self.w.wcs.ctype = ["TLON-CAR", "TLAT-CAR"]
                for i in range(len(self.w.wcs.crval)):
                    if self.w.wcs.crval[i] != 0:
                        self.w.wcs.crval[i] = 0

    def world2pix(self, coord):
        
        '''
        Return pixel coordinates from sky/telescope coordinates
        ''' 
    
        return self.w.all_world2pix(coord, 1)

    def pix2world(self, pixels):
        
        '''
        Function to return world coordinates from pixel coordinates
        '''

        return self.w.all_pix2world(pixels, 1)

    def wcs_proj(self, coord1, coord2, det_num):

        '''
        Wrapper for world2pix to include the possibility of using 
        multiple detectors 
        '''

        if np.size(np.shape(coord1)) != 1:
            self.pixel = np.zeros((det_num, len(coord1[0]), 2), dtype=int)
        else:
            self.pixel = np.zeros((det_num, len(coord1), 2), dtype=int)

        for i in range(det_num):
            coord = np.transpose(np.array([coord1[i], coord2[i]]))
            self.pixel[i,:,:] = (np.floor(self.world2pix(coord))).astype(int)

    def reproject(self, world_original):

        x_min_map = np.floor(np.amin(world_original[:,:,0]))
        y_min_map = np.floor(np.amin(world_original[:,:,1]))

        new_proj = self.w.deepcopy()

        crpix_new = self.w.wcs.crpix-(np.array([x_min_map, y_min_map]))

        new_proj.wcs.crpix = crpix_new

        return new_proj

class mapmaking(object):

    '''
    Class to generate the maps. For more information about the system to be solved
    check Moncelsi et al. 2012
    '''

    def __init__(self, data, weight, polangle, pixelmap, crpix, Ionly=True, number=1, \
                 convolution=False, std=0., cdelt=0.):

        self.data = data               #detector TOD
        self.sigma = 1/weight**2           #weights associated with the detector values
        self.polangle = polangle       #polarization angles of each detector
        self.pixelmap = pixelmap      #Coordinates of each point in the TOD in pixel coordinates

        self.Ionly = Ionly             #Choose if a map only in Intensity
        self.convolution = convolution       #If true a map is convolved with a gaussian
        
        if self.convolution:
            self.std_pixel = std/3600./np.abs(cdelt[0])


    def pointing_matrix_binning(self, param):

        shape_x = int(np.ptp(self.pixelmap[:,:,0])+1)
        shape_y = int(np.ptp(self.pixelmap[:,:,1])+1)
        
        x_min = np.amin(self.pixelmap[:,:,0])
        y_min = np.amin(self.pixelmap[:,:,1])

        points = self.pixelmap.copy()
        points[:,:,0] -= x_min
        points[:,:,1] -= y_min

        temp = np.zeros(int(shape_x*shape_y))

        for i in range(np.shape(points)[0]):
            idx = np.ravel_multi_index(np.flip(points[i].T, axis=0), (shape_y, shape_x))
            temp += np.bincount(idx, weights=param[i], minlength=int(shape_x*shape_y))

        return np.reshape(temp, (shape_y, shape_x)) 

    def binning_map(self):

        maps = []

        I_est = self.pointing_matrix_binning(param=self.data)*self.sigma
        hits = 0.5*self.pointing_matrix_binning(param=np.ones_like(self.data))
        
        Imap = np.zeros_like(I_est)

        if self.Ionly:
                        
            Imap[hits>0] = I_est[hits>0]/hits[hits>0]

            if self.convolution:
                Imap = self.map_convolve(self.std_pixel, Imap)

            maps.append(Imap)

        else:
            try:
                cos = np.cos(2.*self.polangle)
                sin = np.sin(2.*self.polangle)

                Q_est = self.pointing_matrix_binning(param=self.data*cos)*self.sigma
                U_est = self.pointing_matrix_binning(param=self.data*sin)*self.sigma

                c = self.pointing_matrix_binning(param=0.5*cos)*self.sigma
                s = self.pointing_matrix_binning(param=0.5*sin)*self.sigma
                
                c2 = self.pointing_matrix_binning(param=0.5*cos**2)*self.sigma
                
                m = self.pointing_matrix_binning(param=0.5*cos*sin)*self.sigma

                Delta = (c**2*(c2-hits)+2*s*c*m-c2*s**2-\
                         hits*(c2**2+m**2-c2*hits))

                if np.any(Delta[hits>0]==0):
                    raise MapError
                else:
                    A = -(c2**2+m**2-c2*hits)
                    B = c*(c2-hits)+s*m
                    C = c*m-s*c2
                    D = -((c2-hits)*hits+s**2)
                    E = c*s-m*hits
                    F = c2*hits-c**2

                    Qmap = np.zeros_like(I_est)
                    Umap = np.zeros_like(I_est)

                    Imap[hits>0] = (A[hits>0]*I_est[hits>0]+B[hits>0]*Q_est[hits>0]+\
                                    C[hits>0]*U_est[hits>0])/Delta[hits>0]
                    Qmap[hits>0] = (B[hits>0]*I_est[hits>0]+D[hits>0]*Q_est[hits>0]+\
                                    E[hits>0]*U_est[hits>0])/Delta[hits>0]
                    Umap[hits>0] = (C[hits>0]*I_est[hits>0]+E[hits>0]*Q_est[hits>0]+\
                                    F[hits>0]*U_est[hits>0])/Delta[hits>0]

                    if self.convolution:
                        Imap = self.map_convolve(self.std_pixel, Imap)
                        Qmap = self.map_convolve(self.std_pixel, Qmap)
                        Umap = self.map_convolve(self.std_pixel, Umap)

                    maps.append(Imap)
                    maps.append(Qmap)
                    maps.append(Umap)

            except MapError:
                print('The matrix is singular in at least a pixel. Check the input data')
                sys.exit(1)

        return maps

    def map_convolve(self, std, map_value):

        '''
        Function to convolve the maps with a gaussian.
        STD is in pixel values
        '''

        kernel = Gaussian2DKernel(x_stddev=std)

        mask = np.ones_like(map_value)
        mask[np.isfinite(map_value)] = 0

        convolved_map = convolve(map_value, kernel, mask=mask, boundary=None)

        convolved_map[convolved_map==0] = np.nan

        return convolved_map

    