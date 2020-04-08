import numpy as np
from astropy import wcs
from astropy.convolution import Gaussian2DKernel, convolve

class wcs_world():

    '''
    Class to generate a wcs using astropy routines.
    '''

    def __init__(self, ctype, crpix, crdelt, crval, radesys, equinox):

        self.ctype = ctype          #ctype of the map, which projection is used to convert coordinates to pixel numbers
        self.crdelt = crdelt        #cdelt of the map, distance in deg between two close pixels
        self.crpix = crpix          #crpix of the map, central pixel of the map in pixel coordinates
        self.crval = crval          #crval of the map, central pixel of the map in sky/telescope (depending on the system) coordinates
        self.radesys = radesys      #Frame of the observation
        self.equinox = equinox      #Time of the observation

        self.w = wcs.WCS(naxis=2)
        self.w.wcs.crpix = self.crpix
        self.w.wcs.cdelt = self.crdelt
        self.w.wcs.crval = self.crval
        self.w.wcs.radesys = self.radesys
        self.w.wcs.equinox = self.equinox

    def world(self, coord, pang, telcoord):
        
        '''
        Function for creating a wcs projection and a pixel coordinates 
        from sky/telescope coordinates
        '''

        if telcoord is False:
            if self.ctype == 'XY Stage':
                world = np.zeros_like(coord)
                try:
                    world[:,0] = coord[:,0]/(np.amax(coord[:,0]))*360.
                    world[:,1] = coord[:,1]/(np.amax(coord[:,1]))*360.
                except IndexError:
                    world[0,0] = coord[0,0]/(np.amax(coord[0,0]))*360.
                    world[0,1] = coord[0,1]/(np.amax(coord[0,1]))*360.
                self.w.wcs.ctype = ["TLON-CAR", "TLAT-CAR"]
            else:
                if self.ctype == 'RA and DEC':
                    self.w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
                elif self.ctype == 'AZ and EL':
                    self.w.wcs.ctype = ["TLON-ARC", "TLAT-ARC"]
                elif self.ctype == 'CROSS-EL and EL':
                    self.w.wcs.ctype = ["TLON-CAR", "TLAT-CAR"]
                
                world = self.w.all_world2pix(coord, 1)

        else:
            self.w.wcs.ctype = ["TLON-TAN", "TLAT-TAN"]
            world = np.zeros_like(coord)
            px = self.w.wcs.s2p(coord, 1)

            #Use the parallactic angle to rotate the projected plane
            world[:,0] = ((px['imgcrd'][:,0]*np.cos(pang)-px['imgcrd'][:,1]*np.sin(pang))\
                          /self.crdelt[0]+self.crpix[0])
            world[:,1] = ((px['imgcrd'][:,0]*np.sin(pang)+px['imgcrd'][:,1]*np.cos(pang))\
                          /self.crdelt[1]+self.crpix[1])

        return world

    def wcs_proj(self, coord1, coord2, det_num, parang=0, telcoord=False):

        '''
        Function to compute the projection and the pixel coordinates
        '''

        if det_num == 1:

            if np.size(np.shape(coord1)) != 1:
                coord = np.transpose(np.array([coord1[0], coord2[0]]))
            else:
                coord = np.transpose(np.array([coord1, coord2]))

            self.pixel = self.world(coord, parang, telcoord)

        else:
            if np.size(np.shape(coord1)) == 1:
                coord = np.transpose(np.array([coord1, coord2]))
                self.pixel = self.world(coord, parang[0,:], telcoord)
            else:
                self.pixel = np.zeros((np.size(np.shape(self.data)), len(coord1[0]), 2))
                for i in range(np.size(np.shape(self.data))):
                    coord = np.transpose(np.array([coord1[i], coord2[i]]))
                    self.pixel[i,:,:] = self.world(coord, parang[i,:], telcoord)


    def reproject(self, world_original):


        self.w.wcs.ctype = self.ctype


        x_min_map = np.floor(np.amin(world_original[:,0]))
        y_min_map = np.floor(np.amin(world_original[:,1]))

        new_proj = self.w.copy()

        crpix_new = self.w.wcs.crpix+np.array([x_min_map, y_min_map])

        crval_new = self.w.all_pix2world(crpix_new[0], crpix_new[1], 1)

        new_proj.wcs.crval = crval_new

        return new_proj

class mapmaking(object):

    '''
    Class to generate the maps. For more information about the system to be solved
    check Moncelsi et al. 2012
    '''

    def __init__(self, data, weight, polangle, pixelmap, crpix, Ionly=True, number=1, \
                 convolution=False, std=0., cdelt=0.):

        self.data = data               #detector TOD
        self.weight = weight           #weights associated with the detector values
        self.polangle = polangle       #polarization angles of each detector
        self.number = number           #Number of detectors to be mapped
        self.pixelmap = np.floor(pixelmap).astype(int)       #Coordinates of each point in the TOD in pixel coordinates
        self.crpix = crpix             #Coordinates of central point 
        self.Ionly = Ionly             #Choose if a map only in Intensity
        self.convolution = convolution       #If true a map is convolved with a gaussian
        self.std = std                 #Standard deviation of the gaussian for the convolution
        self.cdelt = cdelt             #Pixel size of the map. This is used to compute the std in pixels

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

        convolved_map = convolve(map_value, kernel, nan_treatment='fill', fill_value=0)

        convolved_map[convolved_map==0] = np.nan

        return convolved_map

    