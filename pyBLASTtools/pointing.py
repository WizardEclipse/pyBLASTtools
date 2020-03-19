import numpy as np
import gc
from astropy import wcs

import src.quaternion as quat

class utils(object):

    '''
    class to handle conversion between different coodinates sytem 
    '''

    def __init__(self, coord1, coord2, lst = None, lat = None):

        self.coord1 = coord1              #Array of coord 1 (if RA needs to be in hours)
        self.coord2 = np.radians(coord2)  #Array of coord 2 converted in radians   
        self.lst = lst                    #Local Sideral Time in hours
        self.lat = np.radians(lat)        #Latitude converted in radians

    def ra2ha(self):

        '''
        Return the hour angle in hours given the lst and the ra
        i.e. both lst and ra needs to be in hours
        ''' 

        return self.lst-self.coord1

    def ha2ra(self, hour_angle):

        '''
        Return the right ascension in hours given the lst and hour angles
        i.e. both lst and hour angle needs to be in hours
        '''

        return self.lst - hour_angle

    def radec2azel(self):

        '''
        Function to convert RA and DEC to AZ and EL
        '''
        
        hour_angle = np.radians(self.ra2ha()*15.)

        if isinstance(hour_angle, np.ndarray):
            index, = np.where(hour_angle<0)
            hour_angle[index] += 2*np.pi
        else:
            if hour_angle<0:
                hour_angle +=2*np.pi

        el = np.arcsin(np.sin(self.coord2)*np.sin(self.lat)+\
                       np.cos(self.lat)*np.cos(self.coord2)*np.cos(hour_angle))

        az = np.arccos((np.sin(self.coord2)-np.sin(self.lat)*np.sin(el))/(np.cos(self.lat)*np.cos(el)))

        if isinstance(az, np.ndarray):
            index, = np.where(np.sin(hour_angle)>0)
            az[index] = 2*np.pi - az[index]
        else:
            if np.sin(hour_angle)>0 :
                az = 2*np.pi - az
        return np.degrees(az), np.degrees(el)

    def azel2radec(self):

        '''
        Function to convert AZ and EL to RA and DEC
        '''

        dec = np.arcsin(np.sin(self.coord2)*np.sin(self.lat)+\
                        np.cos(self.lat)*np.cos(self.coord2)*np.cos(np.radians(self.coord1)))


        hour_angle = np.arccos((np.sin(self.coord2)-np.sin(self.lat)*np.sin(dec))/(np.cos(self.lat)*np.cos(dec)))

        index, = np.where(np.sin(np.radians(self.coord1)) > 0)
        hour_angle[index] = 2*np.pi - hour_angle[index]

        ra = self.ha2ra(np.degrees(hour_angle)/15.)*15.

        index, = np.where(ra<0)
        ra[index] += 360.

        return ra, np.degrees(dec)

    def parallactic_angle(self):

        '''
        Compute the parallactic angle which is returned in degrees  
        '''

        hour_angle = np.radians((self.ra2ha())*15)

        if isinstance(hour_angle, np.ndarray):
            try:
                index, = np.where(hour_angle<0)
                hour_angle[index] += 2*np.pi
            except ValueError:
                index, = np.where(hour_angle[0]<0)
                hour_angle[0,index] += 2*np.pi
        else:
            if hour_angle<=0:
                hour_angle += 2*np.pi

        y_pa = np.cos(self.lat)*np.sin(hour_angle)
        x_pa = np.sin(self.lat)*np.cos(self.coord2)-np.cos(hour_angle)*np.cos(self.lat)*np.sin(self.coord2)

        pa = np.arctan2(y_pa, x_pa)

        return np.degrees(pa)

class convert_to_telescope(object):

    '''
    Class to convert from sky equatorial coordinates to telescope coordinates
    '''

    def __init__(self, coord1, coord2, lst, lat):

        self.coord1 = coord1           #RA, needs to be in hours       
        self.coord2 = coord2           #DEC
        self.lst = lst 
        self.lat = lat

    def conversion(self):

        '''
        This function rotates the coordinates projected on the plane using the parallactic angle
        '''
        
        parang = utils(self.coord1, self.coord2, self.lst, self.lat)
        pa = parang.parallactic_angle()

        x_tel = np.radians(self.coord1*15)*np.cos(pa)-np.radians(self.coord2)*np.sin(pa)
        y_tel = np.radians(self.coord2)*np.cos(pa)+np.radians(self.coord1*15)*np.sin(pa)

        return np.degrees(x_tel), np.degrees(y_tel)

class apply_offset(object):

    '''
    Class to apply the offset to different coordinates
    '''

    def __init__(self, coord1, coord2, ctype, xsc_offset, det_offset = np.array([0.,0.]),\
                 lst = None, lat = None):

        self.coord1 = coord1                    #Array of coordinate 1
        self.coord2 = coord2                    #Array of coordinate 2
        self.ctype = ctype                      #Ctype of the map
        self.xsc_offset = xsc_offset            #Offset with respect to star cameras in xEL and EL
        self.det_offset = det_offset            #Offset with respect to the central detector in xEL and EL
        self.lst = lst                          #Local Sideral Time array
        self.lat = lat                          #Latitude array

    def correction(self):

        if self.ctype.lower() == 'ra and dec':

            conv2azel = utils(self.coord1, self.coord2, self.lst, self.lat)

            az, el = conv2azel.radec2azel()

            xEL = np.degrees(np.radians(az)*np.cos(np.radians(el)))
            
            ra_corrected = np.zeros((int(np.size(self.det_offset)/2), len(az)))
            dec_corrected = np.zeros((int(np.size(self.det_offset)/2), len(az)))

            for i in range(int(np.size(self.det_offset)/2)):
                
                quaternion = quat.quaternions()
                xsc_quat = quaternion.eul2quat(self.xsc_offset[0], self.xsc_offset[1], 0)
                det_quat = quaternion.eul2quat(self.det_offset[i,0], self.det_offset[i,1], 0)
                off_quat = quaternion.product(det_quat, xsc_quat)

                xEL_offset, EL_offset, roll_offset = quaternion.quat2eul(off_quat)


                # Verify if the signs are still the same as BLASTPol
                xEL_corrected_temp = xEL-xEL_offset
                EL_corrected_temp = el+EL_offset
                AZ_corrected_temp = np.degrees(np.radians(xEL_corrected_temp)/np.cos(np.radians(el)))

                conv2radec = utils(AZ_corrected_temp, EL_corrected_temp, \
                                   self.lst, self.lat)

                ra_corrected[i,:], dec_corrected[i,:] = conv2radec.azel2radec()

            del xEL_corrected_temp
            del EL_corrected_temp
            del AZ_corrected_temp
            gc.collect()

            return ra_corrected, dec_corrected

        elif self.ctype.lower() == 'az and el':

            el_corrected = np.zeros((int(np.size(self.det_offset)/2), len(self.coord1)))
            az_corrected = np.zeros((int(np.size(self.det_offset)/2), len(self.coord2)))

            for i in range(int(np.size(self.det_offset)/2)):
            
                el_corrected[i, :] = self.coord2+self.xsc_offset[1]+self.det_offset[i, 1]

                az_corrected[i, :] = (self.coord1*np.cos(self.coord2)-self.xsc_offset[i]-\
                                      self.det_offset[i, 0])/np.cos(el_corrected)

            return az_corrected, el_corrected

        else:

            el_corrected = np.zeros((int(np.size(self.det_offset)/2), len(self.coord1)))
            xel_corrected = np.zeros((int(np.size(self.det_offset)/2), len(self.coord2)))

            for i in range(int(np.size(self.det_offset)/2)):

                xel_corrected[i, :] = self.coord1-self.xsc_offset[0]-self.det_offset[i, 0]
                el_corrected[i, :] = self.coord2+self.xsc_offset[1]+self.det_offset[i, 1]


            return xel_corrected,el_corrected

class compute_offset(object):

    def __init__(self, coord1_ref, coord2_ref, map_data, \
                 pixel1_coord, pixel2_coord, wcs_trans, ctype, \
                 lst, lat):

        self.coord1_ref = coord1_ref           #Reference value of the map along the x axis in RA and DEC
        self.coord2_ref = coord2_ref           #Reference value of the map along the y axis in RA and DEC
        self.map_data = map_data               #Maps 
        self.pixel1_coord = pixel1_coord       #Array of the coordinates converted in pixel along the x axis
        self.pixel2_coord = pixel2_coord       #Array of the coordinates converted in pixel along the y axis
        self.wcs_trans = wcs_trans             #WCS transformation 
        self.ctype = ctype                     #Ctype of the map
        self.lst = lst                         #Local Sideral Time
        self.lat = lat                         #Latitude

    def centroid(self, threshold=0.275):

        '''
        For more information about centroid calculation see Shariff, PhD Thesis, 2016
        '''

        maxval = np.max(self.map_data)

        y_max, x_max = np.where(self.map_data == maxval)

        gt_inds = np.where(self.map_data > threshold*maxval)

        weight = np.zeros((self.map_data.shape[0], self.map_data.shape[1]))
        weight[gt_inds] = 1.
        a = self.map_data[gt_inds]
        flux = np.sum(a)

        x_coord_max = np.floor(np.amax(self.pixel1_coord))+1
        x_coord_min = np.floor(np.amin(self.pixel1_coord))

        x_arr = np.arange(x_coord_min, x_coord_max)

        y_coord_max = np.floor(np.amax(self.pixel2_coord))+1
        y_coord_min = np.floor(np.amin(self.pixel2_coord))

        y_arr = np.arange(y_coord_min, y_coord_max)

        xx, yy = np.meshgrid(x_arr, y_arr)
        
        x_c = np.sum(xx*weight*self.map_data)/flux
        y_c = np.sum(yy*weight*self.map_data)/flux

        return np.rint(x_c), np.rint(y_c)
    
    def value(self):

        #Centroid of the map
        x_c, y_c = self.centroid()
               
        if self.ctype.lower() == 'ra and dec':
            map_center = wcs.utils.pixel_to_skycoord(x_c, y_c, self.wcs_trans)

            x_map = map_center.ra.hour
            y_map = map_center.dec.degree

            centroid_conv = utils(x_map, y_map, np.average(self.lst), np.average(self.lat))

            coord1_reference = self.coord1_ref/15.

            az_centr, el_centr = centroid_conv.radec2azel()
            xel_centr = az_centr*np.cos(np.radians(el_centr))

        else:
            map_center = self.wcs_trans.wcs_pix2world(x_c, y_c, 1)
            coord1_reference = self.coord1_ref
            el_centr = y_map
            if self.cytpe.lower() == 'xel and el':
                xel_centr = x_map            
            else:
                xel_centr = x_map/np.cos(np.radians(el_centr))
            

        ref_conv = utils(coord1_reference, self.coord2_ref, np.average(self.lst), \
                         np.average(self.lat))

        az_ref, el_ref = ref_conv.radec2azel()

        xel_ref = az_ref*np.cos(np.radians(el_ref))

        return xel_centr-xel_ref, el_ref+el_centr




        








        

        
        


