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
            self.pixel = np.zeros((det_num, len(coord1[0]), 2))
        else:
            self.pixel = np.zeros((det_num, len(coord1), 2))

        for i in range(det_num):
            coord = np.transpose(np.array([coord1[i], coord2[i]]))
            self.pixel[i,:,:] = np.floor(self.world2pix(coord)).astype(int)

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
        self.number = number           #Number of detectors to be mapped
        self.pixelmap = pixelmap      #Coordinates of each point in the TOD in pixel coordinates
        self.crpix = crpix             #Coordinates of central point 
        self.Ionly = Ionly             #Choose if a map only in Intensity
        self.convolution = convolution       #If true a map is convolved with a gaussian
        self.std = std                 #Standard deviation of the gaussian for the convolution
        self.cdelt = cdelt             #Pixel size of the map. This is used to compute the std in pixels


    def pointing_matrix_binning(self, param):

        shape_x = np.ptp(self.pixelmap[:,:,0])
        shape_y = np.ptp(self.pixelmap[:,:,1])
        
        x_min = np.amin(self.pixelmap[:,:,0])
        y_min = np.amin(self.pixelmap[:,:,1])

        points = self.pixelmap.deepcopy()
        points[:,:,0] -= x_min
        points[:,:,1] -= y_min

        temp = np.zeros(shape_x*shape_y)

        for i in range(np.shape(points)[0]):
            idx = np.ravel_multi_index(points[i], (shape_x, shape_y))
            temp += np.bincount(idx, weights=param[i], minlength=shape_x*shape_y)

        return np.reshape(temp, (shape_x, shape_y)) 

    def binning_map(self):

        maps = []

        I_est = self.pointing_matrix_binning(param=self.data)*self.sigma
        hits = 0.5*self.pointing_matrix_binning(param=np.ones_like(self.data))

        if self.Ionly:            
            Imap = I_est[hits>0]/hits[hits>0]

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

                    Imap = np.zeros_like(I_est)
                    Qmap = np.zeros_like(I_est)
                    Umap = np.zeros_like(I_est)

                    Imap[hits>0] = (A[hits>0]*I_est[hits>0]+B[hits>0]*Q_est[hits>0]+\
                                    C[hits>0]*U_est[hits>0])/Delta[hits>0]
                    Qmap[hits>0] = (B[hits>0]*I_est[hits>0]+D[hits>0]*Q_est[hits>0]+\
                                    E[hits>0]*U_est[hits>0])/Delta[hits>0]
                    Umap[hits>0] = (C[hits>0]*I_est[hits>0]+E[hits>0]*Q_est[hits>0]+\
                                    F[hits>0]*U_est[hits>0])/Delta[hits>0]

                    maps.append(Imap)
                    maps.append(Qmap)
                    maps.append(Umap)

            except MapError:
                print('The matrix is singular in at least a pixel. Check the input data')
                sys.exit(1)

            return maps


    def map2d(self):

        '''
        Function to generate the maps using the pixel coordinates to bin
        '''

        if np.size(np.shape(self.data)) == 1:

            if self.Ionly:
                Imap = self.map_singledetector_Ionly(self.crpix)

                if not self.convolution:
                    return Imap
                else:
                    std_pixel = self.std/3600./np.abs(self.cdelt[0])
                    
                    return self.map_convolve(std_pixel, Imap)
            else:        
                Imap, Qmap, Umap = self.map_singledetector(self.crpix)
                if not self.convolution:
                    return Imap, Qmap, Umap
                else:
                    Imap_con = self.map_convolve(self.std, Imap)
                    Qmap_con = self.map_convolve(self.std, Qmap)
                    Umap_con = self.map_convolve(self.std, Umap)
                    return Imap_con, Qmap_con, Umap_con

        else:
            if self.Ionly:
                Imap = self.map_multidetectors_Ionly(self.crpix)

                if not self.convolution:
                    return Imap
                else:
                    std_pixel = self.std/3600./self.cdelt[0]
                    
                    return self.convolution(std_pixel, Imap)
            else:        
                Imap, Qmap, Umap = self.map_multidetectors(self.crpix)
                if not self.convolution:
                    return Imap, Qmap, Umap
                else:
                    Imap_con = self.convolution(self.std, Imap)
                    Qmap_con = self.convolution(self.std, Qmap)
                    Umap_con = self.convolution(self.std, Umap)
                    return Imap_con, Qmap_con, Umap_con

    def map_param(self, crpix, idxpixel, value=None, noise=None, angle=None):

        '''
        Function to calculate the parameters of the map. Parameters follow the same 
        naming scheme used in the paper
        '''

        if value is None:
            value = self.data.copy()
        if noise is not None:
            sigma = 1/noise**2
        else:
            sigma = 1.
        if np.size(angle) > 1:
            angle = angle.copy()
        else:
            angle = angle*np.ones(np.size(value))

        '''
        sigma is the inverse of the sqared white noise value, so it is 1/n**2
        ''' 
        
        x_map = idxpixel[:,0]   #RA 
        y_map = idxpixel[:,1]   #DEC
        
        if (np.amin(x_map)) <= 0:
            x_map = np.floor(x_map+np.abs(np.amin(x_map)))
        else:
            x_map = np.floor(x_map-np.amin(x_map))
        if (np.amin(y_map)) <= 0:
            y_map = np.floor(y_map+np.abs(np.amin(y_map)))
        else:
            y_map = np.floor(y_map-np.amin(y_map))

        x_len = np.amax(x_map)-np.amin(x_map)+1
        param = x_map+y_map*x_len
        param = param.astype(int)

        flux = value

        cos = np.cos(2.*angle)
        sin = np.sin(2.*angle)

        I_est_flat = np.bincount(param, weights=flux)*sigma
        Q_est_flat = np.bincount(param, weights=flux*cos)*sigma
        U_est_flat = np.bincount(param, weights=flux*sin)*sigma

        N_hits_flat = 0.5*np.bincount(param)*sigma
        c_flat = np.bincount(param, weights=0.5*cos)*sigma
        c2_flat = np.bincount(param, weights=0.5*cos**2)*sigma
        s_flat = np.bincount(param, weights=0.5*sin)*sigma
        s2_flat = N_hits_flat-c2_flat
        m_flat = np.bincount(param, weights=0.5*cos*sin)*sigma

        return I_est_flat, Q_est_flat, U_est_flat, N_hits_flat, c_flat, c2_flat, s_flat, s2_flat, m_flat, param

    def map_singledetector_Ionly(self, crpix, value=None, noise=None, angle=None, idxpixel = None):
        
        '''
        Function to reshape the previous array to create a 2D map for a single detector
        if only I map is requested
        '''

        if value is None:
            value = self.data.copy()
        else:
            value = value

        if idxpixel is None:
            idxpixel = self.pixelmap.copy()
        else:
            idxpixel = idxpixel
        
        if noise is None:
            noise = 1/self.weight**2
        else:
            noise = noise
        
        if angle is None:
            angle = self.polangle
        else:
            angle = angle

        value =self.map_param(crpix=crpix, idxpixel = idxpixel, value=value, noise=noise, angle=angle)

        I_flat = np.zeros(len(value[0]))

        I_flat[np.nonzero(value[0])] = value[0][np.nonzero(value[0])]/value[3][np.nonzero(value[0])]

        x_len = np.amax(idxpixel[:,0])-np.amin(idxpixel[:,0])
        y_len = np.amax(idxpixel[:,1])-np.amin(idxpixel[:,1])

        if len(I_flat) < (x_len+1)*(y_len+1):
            valmax = (x_len+1)*(y_len+1)
            pmax = np.amax(value[-1])
            I_fin = 0.*np.arange(pmax+1, valmax)
            
            I_flat = np.append(I_flat, I_fin)

        I_flat[I_flat==0] = np.nan

        I_pixel = np.reshape(I_flat, (y_len+1,x_len+1))

        return I_pixel

    def map_multidetectors_Ionly(self, crpix):

        Xmin = np.inf
        Xmax = -np.inf
        Ymin = np.inf
        Ymax = -np.inf

        for i in range(self.number):
            if np.size(np.shape(self.pixelmap)) == 2:
                idxpixel = self.pixelmap.copy()
                Xmin, Xmax = np.amin(idxpixel[:, 0]), np.amax(idxpixel[:, 0])
                Ymin, Ymax = np.amin(idxpixel[:, 1]), np.amax(idxpixel[:,1])
                break
            else:
                idxpixel = self.pixelmap[i].copy()
                Xmin = np.amin(np.array([Xmin,np.amin(idxpixel[:, 0])]))
                Xmax = np.amax(np.array([Xmax,np.amax(idxpixel[:, 0])]))
                Ymin = np.amin(np.array([Ymin,np.amin(idxpixel[:, 1])]))
                Ymax = np.amax(np.array([Ymax,np.amax(idxpixel[:, 1])]))
        
        finalmap_num = np.zeros((int(np.abs(Ymax-Ymin)+1), int(np.abs(Xmax-Xmin)+1)))
        finalmap_den = np.zeros((int(np.abs(Ymax-Ymin)+1), int(np.abs(Xmax-Xmin)+1)))

        for i in range(self.number):

            if np.size(np.shape(self.pixelmap)) == 2:
                idxpixel = self.pixelmap.copy()
            else:
                idxpixel = self.pixelmap[i].copy()

            value = self.map_param(crpix=crpix, idxpixel = idxpixel, value=self.data[i], noise=1/self.weight[i], angle=self.polangle[i])

            num_temp_flat = np.zeros(len(value[0]))
            num_temp_flat[np.nonzero(value[0])] = value[0][np.nonzero(value[0])]
            
            den_temp_flat = np.zeros_like(num_temp_flat)
            den_temp_flat[np.nonzero(value[0])] = value[3][np.nonzero(value[0])]

            Xmin_map_temp, Xmax_map_temp = np.amin(idxpixel[:,0]), np.amax(idxpixel[:,0])
            Ymin_map_temp, Ymax_map_temp = np.amin(idxpixel[:,1]), np.amax(idxpixel[:,1])

            index1x = int(Xmin_map_temp-Xmin)
            index2x = int(index1x + np.abs(Xmax_map_temp-Xmin_map_temp))
            index1y = int(Ymin_map_temp-Ymin)
            index2y = int(index1y + np.abs(Ymax_map_temp-Ymin_map_temp))

            x_len = Xmax_map_temp-Xmin_map_temp
            y_len = Ymax_map_temp-Ymin_map_temp

            if len(value[0]) < (x_len+1)*(y_len+1):
                valmax = (x_len+1)*(y_len+1)
                pmax = np.amax(value[-1])
                num_temp_fin = 0.*np.arange(pmax+1, valmax)
                den_temp_fin = np.ones(np.abs(pmax+1-valmax))
                
                temp_map_num_flat = np.append(num_temp_flat, num_temp_fin)
                temp_map_den_flat = np.append(den_temp_flat, den_temp_fin)

            temp_map_num = np.reshape(temp_map_num_flat, (y_len+1,x_len+1))
            temp_map_den = np.reshape(temp_map_den_flat, (y_len+1,x_len+1))

            finalmap_num[index1y:index2y+1,index1x:index2x+1] += temp_map_num
            finalmap_den[index1y:index2y+1,index1x:index2x+1] += temp_map_den

        finalmap = finalmap_num/finalmap_den

        finalmap[finalmap==0] = np.nan

        return finalmap

    def map_singledetector(self, crpix, value=None, sigma=None, angle=None, idxpixel=None):

        '''
        Function to reshape the previous array to create a 2D map for a single detector
        if also polarization maps are requested
        '''

        if idxpixel is None:
            idxpixel = self.pixelmap.copy()
        else:
            idxpixel = idxpixel

        (I_est_flat, Q_est_flat, U_est_flat, N_hits_flat, \
         c_flat, c2_flat, s_flat, s2_flat, m_flat, param) = self.map_param(crpix=crpix, idxpixel=idxpixel, value=value, \
                                                                           noise=1/self.weight**2,angle=self.polangle)

        Delta = (c_flat**2*(c2_flat-N_hits_flat)+2*s_flat*c_flat*m_flat-c2_flat*s_flat**2-\
                 N_hits_flat*(c2_flat**2+m_flat**2-c2_flat*N_hits_flat))
        A = -(c2_flat**2+m_flat**2-c2_flat*N_hits_flat)
        B = c_flat*(c2_flat-N_hits_flat)+s_flat*m_flat
        C = c_flat*m_flat-s_flat*c2_flat
        D = -((c2_flat-N_hits_flat)*N_hits_flat+s_flat**2)
        E = c_flat*s_flat-m_flat*N_hits_flat
        F = c2_flat*N_hits_flat-c_flat**2

        I_pixel_flat = np.zeros(len(I_est_flat))
        Q_pixel_flat = np.zeros(len(Q_est_flat))
        U_pixel_flat = np.zeros(len(U_est_flat))

        index, = np.where(np.abs(Delta)>0.)
        
        I_pixel_flat[index] = (A[index]*I_est_flat[index]+B[index]*Q_est_flat[index]+\
                               C[index]*U_est_flat[index])/Delta[index]
        Q_pixel_flat[index] = (B[index]*I_est_flat[index]+D[index]*Q_est_flat[index]+\
                               E[index]*U_est_flat[index])/Delta[index]
        U_pixel_flat[index] = (C[index]*I_est_flat[index]+E[index]*Q_est_flat[index]+\
                               F[index]*U_est_flat[index])/Delta[index]

        x_len = np.amax(idxpixel[:,0])-np.amin(idxpixel[:,0])
        y_len = np.amax(idxpixel[:,1])-np.amin(idxpixel[:,1])

        if len(I_est_flat) < (x_len+1)*(y_len+1):
            valmax = (x_len+1)*(y_len+1)
            pmax = np.amax(param)
            I_fin = 0.*np.arange(pmax+1, valmax)
            Q_fin = 0.*np.arange(pmax+1, valmax)
            U_fin = 0.*np.arange(pmax+1, valmax)
            
            I_pixel_flat = np.append(I_pixel_flat, I_fin)
            Q_pixel_flat = np.append(Q_pixel_flat, Q_fin)
            U_pixel_flat = np.append(U_pixel_flat, U_fin)

        I_pixel_flat[I_pixel_flat==0] = np.nan
        Q_pixel_flat[Q_pixel_flat==0] = np.nan
        U_pixel_flat[U_pixel_flat==0] = np.nan
        
        I_pixel = np.reshape(I_pixel_flat, (y_len+1,x_len+1))
        Q_pixel = np.reshape(Q_pixel_flat, (y_len+1,x_len+1))
        U_pixel = np.reshape(U_pixel_flat, (y_len+1,x_len+1))

        return I_pixel, Q_pixel, U_pixel

    def map_multidetectors(self, crpix):


        Xmin = np.inf
        Xmax = -np.inf
        Ymin = np.inf
        Ymax = -np.inf

        for i in range(self.number):
            if np.size(np.shape(self.pixelmap)) == 2:
                idxpixel = self.pixelmap.copy()
                Xmin, Xmax = np.amin(idxpixel[:, 0]), np.amax(idxpixel[:, 0])
                Ymin, Ymax = np.amin(idxpixel[:, 1]), np.amax(idxpixel[:,1])
                break
            else:
                idxpixel = self.pixelmap[i].copy()
                Xmin = np.amin(np.array([Xmin,np.amin(idxpixel[:, 0])]))
                Xmax = np.amax(np.array([Xmax,np.amax(idxpixel[:, 0])]))
                Ymin = np.amin(np.array([Ymin,np.amin(idxpixel[:, 1])]))
                Ymax = np.amax(np.array([Ymax,np.amax(idxpixel[:, 1])]))
        
        finalmap_I_est = np.zeros((int(np.abs(Ymax-Ymin)+1), int(np.abs(Xmax-Xmin)+1)))
        finalmap_Q_est = np.zeros((int(np.abs(Ymax-Ymin)+1), int(np.abs(Xmax-Xmin)+1)))
        finalmap_U_est = np.zeros((int(np.abs(Ymax-Ymin)+1), int(np.abs(Xmax-Xmin)+1)))
        finalmap_N_hits = np.zeros((int(np.abs(Ymax-Ymin)+1), int(np.abs(Xmax-Xmin)+1)))
        finalmap_c = np.zeros((int(np.abs(Ymax-Ymin)+1), int(np.abs(Xmax-Xmin)+1)))
        finalmap_c2 = np.zeros((int(np.abs(Ymax-Ymin)+1), int(np.abs(Xmax-Xmin)+1)))
        finalmap_s = np.zeros((int(np.abs(Ymax-Ymin)+1), int(np.abs(Xmax-Xmin)+1)))
        finalmap_s2 = np.zeros((int(np.abs(Ymax-Ymin)+1), int(np.abs(Xmax-Xmin)+1)))
        finalmap_m = np.zeros((int(np.abs(Ymax-Ymin)+1), int(np.abs(Xmax-Xmin)+1)))
        finalmap_I = np.zeros((int(np.abs(Ymax-Ymin)+1), int(np.abs(Xmax-Xmin)+1)))
        finalmap_Q = np.zeros((int(np.abs(Ymax-Ymin)+1), int(np.abs(Xmax-Xmin)+1)))
        finalmap_U = np.zeros((int(np.abs(Ymax-Ymin)+1), int(np.abs(Xmax-Xmin)+1)))

        for i in range(self.number):
            if np.size(np.shape(self.pixelmap)) == 2:
                idxpixel = self.pixelmap.copy()
            else:
                idxpixel = self.pixelmap[i].copy()

            (I_est_flat, Q_est_flat, U_est_flat, N_hits_flat, \
             c_flat, c2_flat, s_flat, s2_flat, m_flat, param) = self.map_param(crpix=crpix, idxpixel=idxpixel, value=self.data[i], \
                                                                               noise=1/self.weight[i],angle=self.polangle[i])

            Xmin_map_temp, Xmax_map_temp = np.amin(idxpixel[:,0]), np.amax(idxpixel[:,0])
            Ymin_map_temp, Ymax_map_temp = np.amin(idxpixel[:,1]), np.amax(idxpixel[:,1])

            index1x = int(Xmin_map_temp-Xmin)
            index2x = int(index1x + np.abs(Xmax_map_temp-Xmin_map_temp))
            index1y = int(Ymin_map_temp-Ymin)
            index2y = int(index1y + np.abs(Ymax_map_temp-Ymin_map_temp))

            x_len = Xmax_map_temp-Xmin_map_temp
            y_len = Ymax_map_temp-Ymin_map_temp

            if len(I_est_flat) < (x_len+1)*(y_len+1):
                valmax = (x_len+1)*(y_len+1)
                pmax = np.amax(param)
                I_num_fin = 0.*np.arange(pmax+1, valmax)
                Q_num_fin = 0.*np.arange(pmax+1, valmax)
                U_num_fin = 0.*np.arange(pmax+1, valmax)
                N_hits_fin = 0.*np.arange(pmax+1, valmax)
                c_fin = 0.*np.arange(pmax+1, valmax)
                c2_fin = 0.*np.arange(pmax+1, valmax)
                s_fin = 0.*np.arange(pmax+1, valmax)
                s2_fin = 0.*np.arange(pmax+1, valmax)
                m_fin = 0.*np.arange(pmax+1, valmax)
                
                I_est_flat = np.append(I_est_flat, I_num_fin)
                Q_est_flat = np.append(Q_est_flat, Q_num_fin)
                U_est_flat = np.append(U_est_flat, U_num_fin)
                N_hits_flat = np.append(N_hits_flat, N_hits_fin)
                c_flat = np.append(c_flat, c_fin)
                c2_flat = np.append(c2_flat, c2_fin)
                s_flat = np.append(s_flat, s_fin)
                s2_flat = np.append(s2_flat, s2_fin)
                m_flat = np.append(m_flat, m_fin)

            I_est = np.reshape(I_est_flat, (y_len+1,x_len+1))
            Q_est = np.reshape(Q_est_flat, (y_len+1,x_len+1))
            U_est = np.reshape(U_est_flat, (y_len+1,x_len+1))
            N_hits_est = np.reshape(N_hits_flat, (y_len+1,x_len+1))
            c_est = np.reshape(c_flat, (y_len+1,x_len+1))
            c2_est = np.reshape(c2_flat, (y_len+1,x_len+1))
            s_est = np.reshape(s_flat, (y_len+1,x_len+1))
            s2_est = np.reshape(s2_flat, (y_len+1,x_len+1))
            m_est = np.reshape(m_flat, (y_len+1,x_len+1))


            finalmap_I_est[index1y:index2y+1,index1x:index2x+1] += I_est
            finalmap_Q_est[index1y:index2y+1,index1x:index2x+1] += Q_est
            finalmap_U_est[index1y:index2y+1,index1x:index2x+1] += U_est
            finalmap_N_hits[index1y:index2y+1,index1x:index2x+1] += N_hits_est
            finalmap_c[index1y:index2y+1,index1x:index2x+1] += c_est
            finalmap_c2[index1y:index2y+1,index1x:index2x+1] += c2_est
            finalmap_s[index1y:index2y+1,index1x:index2x+1] += s_est
            finalmap_s2[index1y:index2y+1,index1x:index2x+1] += s2_est
            finalmap_m[index1y:index2y+1,index1x:index2x+1] += m_est


        Delta = (c_flat**2*(c2_flat-N_hits_flat)+2*s_flat*c_flat*m_flat-c2_flat*s_flat**2-\
                 N_hits_flat*(c2_flat**2+m_flat**2-c2_flat*N_hits_flat))
        A = -(c2_flat**2+m_flat**2-c2_flat*N_hits_flat)
        B = c_flat*(c2_flat-N_hits_flat)+s_flat*m_flat
        C = c_flat*m_flat-s_flat*c2_flat
        D = -((c2_flat-N_hits_flat)*N_hits_flat+s_flat**2)
        E = c_flat*s_flat-m_flat*N_hits_flat
        F = c2_flat*N_hits_flat-c_flat**2

        index, = np.where(np.abs(Delta)>0.)

        finalmap_I[index] = (A[index]*finalmap_I_est[index]+B[index]*finalmap_Q_est[index]+\
                             C[index]*finalmap_U_est[index])/Delta[index]
        finalmap_Q[index] = (B[index]*finalmap_I_est[index]+D[index]*finalmap_Q_est[index]+\
                             E[index]*finalmap_U_est[index])/Delta[index]
        finalmap_U[index] = (C[index]*finalmap_I_est[index]+E[index]*finalmap_Q_est[index]+\
                             F[index]*finalmap_U_est[index])/Delta[index]

        finalmap_I[finalmap_I==0] = np.nan
        finalmap_Q[finalmap_Q==0] = np.nan
        finalmap_U[finalmap_U==0] = np.nan

        return finalmap_I, finalmap_Q, finalmap_U

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

    