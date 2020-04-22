import numpy as np
import gc
from astropy import wcs
import astropy.units as u
from astropy.coordinates import EarthLocation, AltAz, FK5, ICRS, SkyCoord
from astropy.time import Time

import pyBLASTtools.quaternion as quat

class utils(object):

    '''
    class to handle conversion between different coodinates sytem and compute 
    useful astronomical quantities
    '''

    def __init__(self, lon, lat, height, time, ra=None, dec=None, az=None, alt=None, \
                 radec_frame='current', coord1_unit=u.degree, coord2_unit=u.degree, lon_unit=u.degree, \
                 lat_unit=u.degree, height_unit=u.m, time_unit=u.s):

        if ra is not None and dec is not None:
            self.system = 'celestial'
            coord1 = ra
            coord2 = dec

        if az is not None and alt is not None:
            self.system = 'horizontal'
            coord1 = az
            coord2 = alt

        if isinstance(coord1, u.quantity.Quantity):
            self.coord1 = coord1
        else:
            self.coord1 = coord1*coord1_unit

        if isinstance(coord2, u.quantity.Quantity):
            self.coord2 = coord2
        else:
            self.coord2 = coord2*coord2_unit

        if isinstance(lon, u.quantity.Quantity):
            self.lon = lon
        else:
            self.lon = lon*lon_unit

        if isinstance(lat, u.quantity.Quantity):
            self.lat = lat
        else:
            self.lat = lat*lat_unit

        if isinstance(height, u.quantity.Quantity):
            self.height = height
        else:
            self.height = height*height_unit

        if isinstance(time, u.quantity.Quantity):
            self.time = Time(time, format='unix', scale='utc')
        else:
            self.time = Time(time*time_unit, format='unix', scale='utc')

        if radec_frame.lower() == 'icrs':
            self.radec_frame = ICRS()
        elif radec_frame.lower() == 'j2000':
            self.radec_frame = FK5(equinox='j2000')
        elif radec_frame.lower() == 'current':
            time_epoch = np.mean(time)
            self.radec_frame = FK5(equinox=Time(time_epoch*time_unit, format='unix', scale='utc'))
        else:
            self.radec_frame = radec_frame

        self.location = EarthLocation(lon=self.lon, lat=self.lat, height=self.height)

        self.altaz_frame = AltAz(obstime = self.time, location = self.location)

        if self.system == 'celestial':
            self.coordinates = SkyCoord(self.coord1, self.coord2, frame=self.radec_frame)
        elif self.system == 'horizontal':
            self.coordinates = SkyCoord(self.coord1, self.coord2, frame=self.altaz_frame)

    def horizontal2sky(self):

        '''
        Convert horizontal coordinates to sky coordinates using astropy routines
        '''

        temp = self.coordinates.transform_to(self.radec_frame)

        return temp.ra.deg, temp.dec.deg

    def sky2horizontal(self):

        '''
        Convert sky coordinates to horizontal coordinates using astropy routines
        '''

        temp = self.coordinates.transform_to(self.altaz_frame)

        return temp.az.deg, temp.alt.deg

    def parallactic_angle(self):
        
        '''
        Compute the parallactic angle 
        '''

        if self.system == 'celestial':
            coord = self.coordinates
        elif self.system == 'horizontal':
            coord_temp = self.horizontal2sky()
            coord = SkyCoord(coord_temp[0], coord_temp[1], unit=u.degree, frame=self.radec_frame)

        LST = self.time.sidereal_time('mean', longitude=self.location.lon)
        H = (LST - coord.ra).radian
        q = np.arctan2(np.sin(H),
                       (np.tan(self.location.lat.radian)*np.cos(coord.dec.radian)-\
                        np.sin(coord.dec.radian)*np.cos(H)))

        return np.degrees(q) 

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
        pa = np.radians(parang.parallactic_angle())

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

        return x_c, y_c
    
    def value(self):

        #Centroid of the map
        x_c, y_c = self.centroid()
               
        if self.ctype.lower() == 'ra and dec':
            map_center = wcs.utils.pixel_to_skycoord(np.rint(x_c), np.rint(y_c), self.wcs_trans)
            
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


        








        

        
        


