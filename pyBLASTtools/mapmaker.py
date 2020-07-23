import numpy as np
from astropy import wcs
from astropy.convolution import Gaussian2DKernel, convolve

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
            try:
                coord = np.transpose(np.array([coord1[i], coord2[i]]))
            except IndexError:
                coord = np.transpose(np.array([coord1[0], coord2[0]]))
            self.pixel[i,:,:] = (np.rint(self.world2pix(coord))).astype(int)

    def reproject(self, world_original):

        x_min_map = np.rint(np.amin(world_original[:,:,0]))
        y_min_map = np.rint(np.amin(world_original[:,:,1]))

        new_proj = self.w.deepcopy()

        crpix_new = self.w.wcs.crpix-(np.array([x_min_map, y_min_map]))

        new_proj.wcs.crpix = crpix_new

        return new_proj

class mapmaking(object):

    '''
    Class to generate the maps. For more information about the system to be solved
    check Moncelsi et al. 2012
    '''

    def __init__(self, data, weight, polangle, pixelmap, crpix, det_idx=None, Ionly=True, number=1, \
                 convolution=False, std=0., cdelt=0.):

        self.data = data                     #detector TOD
        self.sigma = 1/weight**2             #weights associated with the detector values
        self.polangle = polangle             #polarization angles of each detector
        self.pixelmap = pixelmap.copy()      #Coordinates of each point in the TOD in pixel coordinates

        if det_idx is None:
            det_idx = np.arange(np.shape(self.data)[0])

        self.det_idx = det_idx               #Detector index, used for not coadded map to label the dictionary 

        self.Ionly = Ionly                   #Choose if a map only in Intensity
        self.convolution = convolution       #If true a map is convolved with a gaussian
        
        if self.convolution:
            self.std_pixel = std/3600./np.abs(cdelt[0])

        self.map_shape_x = int(np.ptp(self.pixelmap[:,:,0])+1)
        self.map_shape_y = int(np.ptp(self.pixelmap[:,:,1])+1)

        x_min = np.amin(self.pixelmap[:,:,0])
        y_min = np.amin(self.pixelmap[:,:,1])

        self.pixelmap[:,:,0] -= x_min
        self.pixelmap[:,:,1] -= y_min

        try:
            from mpi4py import MPI
            self.mpi = MPI
            self.mpi_implementation = True
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.nprocs = self.comm.Get_size()
        except ImportError:
            self.mpi_implementation = False
            self.rank = 0
            self.nprocs = 1

    def pointing_matrix_binning(self, param, coord):
        
        idx = np.ravel_multi_index(np.flip(coord.T, axis=0), (self.map_shape_y, self.map_shape_x))
        temp = np.bincount(idx, weights=param, minlength=int(self.map_shape_x*self.map_shape_y))

        return np.reshape(temp, (self.map_shape_y, self.map_shape_x))

    def polarization_binning(self, I, Q, U, hits, c, s, c2, m):

        Delta = (c**2*(c2-hits)+2*s*c*m-c2*s**2-\
                 hits*(c2**2+m**2-c2*hits))

        if np.any(Delta[hits>0]==0):
            return None, None, None
        else:
            A = -(c2**2+m**2-c2*hits)
            B = c*(c2-hits)+s*m
            C = c*m-s*c2
            D = -((c2-hits)*hits+s**2)
            E = c*s-m*hits
            F = c2*hits-c**2

            Imap = np.zeros_like(I)
            Qmap = np.zeros_like(I)
            Umap = np.zeros_like(I)

            Imap[hits>0] = (A[hits>0]*I[hits>0]+B[hits>0]*Q[hits>0]+\
                            C[hits>0]*U[hits>0])/Delta[hits>0]
            Qmap[hits>0] = (B[hits>0]*I[hits>0]+D[hits>0]*Q[hits>0]+\
                            E[hits>0]*U[hits>0])/Delta[hits>0]
            Umap[hits>0] = (C[hits>0]*I[hits>0]+E[hits>0]*Q[hits>0]+\
                            F[hits>0]*U[hits>0])/Delta[hits>0]

            return Imap, Qmap, Umap

    def binning_map(self, coadd=False):

        maps = {}

        I_est = np.zeros((int(self.map_shape_y),int(self.map_shape_x)))
        hits = np.zeros((int(self.map_shape_y),int(self.map_shape_x)))

        if not self.Ionly:
            Q_est = np.zeros((int(self.map_shape_y),int(self.map_shape_x)))
            U_est = np.zeros((int(self.map_shape_y),int(self.map_shape_x)))

            c_est = np.zeros((int(self.map_shape_y),int(self.map_shape_x)))
            s_est = np.zeros((int(self.map_shape_y),int(self.map_shape_x)))

            c2_est = np.zeros((int(self.map_shape_y),int(self.map_shape_x)))

            m_est = np.zeros((int(self.map_shape_y),int(self.map_shape_x)))


        if self.mpi_implementation:

            if coadd:
                I_est_total = np.zeros((int(self.map_shape_y),int(self.map_shape_x)))
                hits_total = np.zeros((int(self.map_shape_y),int(self.map_shape_x)))

                if not self.Ionly:

                    Q_est_total = np.zeros((int(self.map_shape_y),int(self.map_shape_x)))
                    U_est_total = np.zeros((int(self.map_shape_y),int(self.map_shape_x)))

                    c_est_total = np.zeros((int(self.map_shape_y),int(self.map_shape_x)))
                    s_est_total = np.zeros((int(self.map_shape_y),int(self.map_shape_x)))

                    c2_est_total = np.zeros((int(self.map_shape_y),int(self.map_shape_x)))

                    m_est_total = np.zeros((int(self.map_shape_y),int(self.map_shape_x)))
            
            else:
                I_mpi = []

                if not self.Ionly:
                    Q_mpi = []
                    U_mpi = []

        if not coadd:
            maps['I'] = {}

            if not self.Ionly:
                maps['Q'] = {}
                maps['U'] = {}

        for i in range(self.rank,np.shape(self.pixelmap)[0],self.nprocs):

            I_temp = self.pointing_matrix_binning(param=self.data[i], coord=self.pixelmap[i])
            hits_temp = 0.5*self.pointing_matrix_binning(param=np.ones_like(self.data[i]),\
                                                         coord=self.pixelmap[i])

            if not self.Ionly:
                cos = np.cos(2.*self.polangle)
                sin = np.sin(2.*self.polangle)

                Q_temp = self.pointing_matrix_binning(param=self.data[i]*cos, \
                                                      coord=self.pixelmap[i])*self.sigma
                U_temp = self.pointing_matrix_binning(param=self.data[i]*sin, \
                                                      coord=self.pixelmap[i])*self.sigma

                c_temp = self.pointing_matrix_binning(param=0.5*cos, coord=self.pixelmap[i])*self.sigma
                s_temp = self.pointing_matrix_binning(param=0.5*sin, coord=self.pixelmap[i])*self.sigma
                
                c2_temp = self.pointing_matrix_binning(param=0.5*cos**2, coord=self.pixelmap[i])*self.sigma
                
                m_temp = self.pointing_matrix_binning(param=0.5*cos*sin, coord=self.pixelmap[i])*self.sigma

            if coadd:
                I_est += I_temp
                hits += hits_temp

                if not self.Ionly:

                    Q_est += Q_temp
                    U_est += U_temp

                    c_est += c_temp
                    s_est += s_temp

                    c2_est += c2_temp

                    m_est += m_temp

            else:

                if self.Ionly:
                    Imap = np.zeros((int(self.map_shape_y),int(self.map_shape_x)))
                    Imap[hits_temp>0] = I_temp[hits_temp>0]/hits_temp[hits_temp>0]
                
                else:
                    Imap, Qmap, Umap = self.polarization_binning(I_temp, Q_temp, \
                                                                 U_temp, hits_temp, \
                                                                 c_temp, s_temp,\
                                                                 c2_temp, m_temp)

                if self.convolution:
                    Imap = self.map_convolve(self.std_pixel, Imap)
                    if not self.Ionly:
                        Qmap = self.map_convolve(self.std_pixel, Qmap)
                        Umap = self.map_convolve(self.std_pixel, Umap)
                
                if self.mpi_implementation:
                    temp = {}
                    temp['det_'+str(int(self.det_idx[i]))] = Imap
                    I_mpi.append(temp)

                    if not self.Ionly:
                        temp = {}
                        temp['det_'+str(int(self.det_idx[i]))] = Qmap
                        Q_mpi.append(temp)

                        temp = {}
                        temp['det_'+str(int(self.det_idx[i]))] = Umap
                        U_mpi.append(temp)

                else:
                    maps['I']['det_'+str(int(self.det_idx[i]))] = Imap
                    if not self.Ionly:
                        maps['Q']['det_'+str(int(self.det_idx[i]))] = Qmap
                        maps['U']['det_'+str(int(self.det_idx[i]))] = Umap

        if coadd:
            if self.mpi_implementation:
                self.comm.Allreduce([I_est, self.mpi.DOUBLE], [I_est_total, self.mpi.DOUBLE], \
                                    op=self.mpi.SUM)
                self.comm.Allreduce([hits, self.mpi.DOUBLE], [hits_total, self.mpi.DOUBLE], \
                                    op=self.mpi.SUM)

                if self.Ionly:
                    Imap = np.zeros_like(I_est_total)

                    Imap[hits_total>0] = I_est_total[hits_total>0]/hits_total[hits_total>0]

                    maps['I'] = Imap

                else:
                    self.comm.Allreduce([Q_est, self.mpi.DOUBLE], [Q_est_total, self.mpi.DOUBLE], \
                                        op=self.mpi.SUM)
                    self.comm.Allreduce([U_est, self.mpi.DOUBLE], [U_est_total, self.mpi.DOUBLE], \
                                        op=self.mpi.SUM)
                    
                    self.comm.Allreduce([c_est, self.mpi.DOUBLE], [c_est_total, self.mpi.DOUBLE], \
                                        op=self.mpi.SUM)
                    self.comm.Allreduce([s_est, self.mpi.DOUBLE], [s_est_total, self.mpi.DOUBLE], \
                                        op=self.mpi.SUM)

                    self.comm.Allreduce([c2_est, self.mpi.DOUBLE], [c2_est_total, self.mpi.DOUBLE], \
                                        op=self.mpi.SUM)

                    self.comm.Allreduce([m_est, self.mpi.DOUBLE], [m_est_total, self.mpi.DOUBLE], \
                                        op=self.mpi.SUM)

                    
                    maps['I'], maps['Q'], maps['U'] = self.polarization_binning(I_est_total, Q_est_total, \
                                                                                U_est_total, hits_total, \
                                                                                c_est_total, s_est_total,\
                                                                                c2_est_total, m_est_total)

            else:
                if self.Ionly:
                    Imap = np.zeros_like(I_est)
                    Imap[hits>0] = I_est[hits>0]/hits_total[hits>0]
                    maps['I'] = Imap

                else:
                    maps['I'], maps['Q'], maps['U'] = self.polarization_binning(I_est, Q_est, \
                                                                                U_est, hits, \
                                                                                c_est, s_est,\
                                                                                c2_est, m_est)

            if self.convolution:
                maps['I'] = self.map_convolve(self.std_pixel, maps['I'])
                if not self.Ionly:
                    maps['Q'] = self.map_convolve(self.std_pixel, maps['Q'])
                    maps['U'] = self.map_convolve(self.std_pixel, maps['U'])

        else:
            if self.mpi_implementation:
                self.comm.Barrier()
                
                dataI = self.comm.allgather(I_mpi)

                if not self.Ionly:
                    dataQ = self.comm.allgather(Q_mpi)
                    dataU = self.comm.allgather(U_mpi)
                
                if self.rank == 0:
                    for h in range(len(dataI)):
                        for j in range(len(dataI[h])):
                            maps['I'].update(dataI[h][j])

                            if not self.Ionly:
                                maps['Q'].update(dataQ[h][j])
                                maps['U'].update(dataU[h][j])

                maps = self.comm.bcast(maps, root=0)

        return maps

    def median_maps(self, maps):

        median = {}

        if len(maps) == 1:

            median['I'] = np.median(list(maps.values()), axis=0)

        else:

            median['I'] = np.median(list(maps['I'].values()), axis=0)
            median['Q'] = np.median(list(maps['Q'].values()), axis=0)
            median['U'] = np.median(list(maps['U'].values()), axis=0)

        return median

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

    