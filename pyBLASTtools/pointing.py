import numpy as np
import astropy.units as u
from astropy.coordinates import EarthLocation, AltAz, FK5, ICRS, SkyCoord
from astropy.time import Time, core

import pyBLASTtools.quaternion as quat
import pyBLASTtools.beam as beam

class utils(object):

    '''
    class to handle conversion between different coodinates sytem and compute 
    useful astronomical quantities
    '''

    def __init__(self, **kwargs):

        '''
        Possible arguments:
        - lon: Longitude (in degrees otherwise specify lon_unit as an astropy unit)
        - lat: Latitude (in degrees otherwise specify lat_unit as an astropy unit)
        - height: Altitude (in meters otherwise specify height_unit as an astropy unit)
        - time: time of the observation in unix format and in seconds or as an astropy.time.core.Time object
        - ra: Right ascension (in degrees otherwise specify coord1_unit as an astropy unit)
        - dec: Declination (in degrees otherwise specify coord2_unit as an astropy unit)
        - az: Azimuth (in degrees otherwise specify coord1_unit as an astropy unit)
        - alt: Elevation (in degrees otherwise specify coord2_unit as an astropy unit)
        - radec_frame: Frame of the Equatorial system, ICRS, current or J2000. Default is current if
                       a time is defined, otherwise is J2000
        '''

        ra = kwargs.get('ra')
        dec = kwargs.get('dec')
        if ra is not None and dec is not None:
            self.system = 'celestial'
            coord1 = ra
            coord2 = dec

        az = kwargs.get('az')
        alt = kwargs.get('alt')
        if az is not None and alt is not None:
            self.system = 'horizontal'
            coord1 = az
            coord2 = alt

        coord1_unit = kwargs.get('coord1_unit', u.degree)
        if isinstance(coord1, u.quantity.Quantity):
            self.coord1 = coord1
        else:
            self.coord1 = coord1*coord1_unit

        coord2_unit = kwargs.get('coord2_unit', u.degree)
        if isinstance(coord2, u.quantity.Quantity):
            self.coord2 = coord2
        else:
            self.coord2 = coord2*coord2_unit

        lon = kwargs.get('lon')
        lon_unit = kwargs.get('lon_unit', u.degree)
        if isinstance(lon, u.quantity.Quantity):
            self.lon = lon
        else:
            self.lon = lon*lon_unit

        lat = kwargs.get('lat')
        lat_unit = kwargs.get('lat_unit', u.degree)
        if isinstance(lat, u.quantity.Quantity):
            self.lat = lat
        else:
            self.lat = lat*lat_unit

        height = kwargs.get('height', 0.)
        height_unit = kwargs.get('height_unit', u.meter)
        if isinstance(height, u.quantity.Quantity):
            self.height = height
        else:
            self.height = height*height_unit

        time = kwargs.get('time')
        if isinstance(time, core.Time):
            self.time = Time
        else:
            if time is not None:
                self.time = Time(time*u.s, format='unix', scale='utc')
            else:
                self.time = Time('J2000')

        if time is not None:
            radec_frame = kwargs.get('radec_frame', 'current')
        else:
            radec_frame = 'j2000'

        if radec_frame.lower() == 'icrs':
            self.radec_frame = ICRS()
        elif radec_frame.lower() == 'j2000':
            self.radec_frame = FK5(equinox='j2000')
        elif radec_frame.lower() == 'current':
            time_epoch = np.mean(self.time.unix)
            self.radec_frame = FK5(equinox=Time(time_epoch, format='unix', scale='utc'))
        else:
            self.radec_frame = radec_frame

        self.location = EarthLocation(lon=self.lon, lat=self.lat, height=self.height)

        self.altaz_frame = AltAz(obstime = self.time, location = self.location)

        self.coordinates = []

        if self.system == 'celestial':
            self.coordinates = SkyCoord(self.coord1, self.coord2, frame=self.radec_frame)
        elif self.system == 'horizontal':
            self.coordinates = SkyCoord(self.coord1, self.coord2, frame=self.altaz_frame)

    def horizontal2sky(self):

        '''
        Convert horizontal coordinates to sky coordinates using astropy routines
        '''

        self.radec_coord = self.coordinates.transform_to(self.radec_frame)

        return self.radec_coord.ra.deg, self.radec_coord.dec.deg

    def sky2horizontal(self):

        '''
        Convert sky coordinates to horizontal coordinates using astropy routines
        '''

        self.altaz_coord = self.coordinates.transform_to(self.altaz_frame)

        return self.altaz_coord.az.deg, self.altaz_coord.alt.deg

    def parallactic_angle(self):
        
        '''
        Compute the parallactic angle 
        '''

        if self.system == 'celestial':
            coord = self.coordinates
        elif self.system == 'horizontal':
            try:
                coord = self.radec_coord
            except AttributeError:
                coord_temp = self.horizontal2sky()
                coord = SkyCoord(coord_temp[0], coord_temp[1], unit=u.degree, frame=self.radec_frame)

        LST = self.time.sidereal_time('mean', longitude=self.location.lon)
        H = (LST - coord.ra).radian
        q = np.arctan2(np.sin(H)*np.cos(self.location.lat.radian),
                       (np.sin(self.location.lat.radian)*np.cos(coord.dec.radian)-\
                        np.sin(coord.dec.radian)*np.cos(self.location.lat.radian)*np.cos(H)))

        return np.degrees(q) 

class convert_to_telescope():

    '''
    Class to convert from sky equatorial coordinates to telescope coordinates using a tangential projection
    '''

    def __init__(self, coord1, coord2, **kwargs):

        self.coord1 = coord1           #RA in degrees       
        self.coord2 = coord2           #DEC in degrees
        
        self.pa = kwargs.get('pa')

        if self.pa is None:
            self.lon = kwargs.get('lon') 
            self.lat = kwargs.get('lat')
            self.time = kwargs.get('time')
            
            if self.lon is not None and self.lat is not None and self.time is not None:

                parang = utils(ra=self.coord1, dec=self.coord2, lon=self.lon, lat=self.lat, time=self.time)
                self.pa = np.radians(parang.parallactic_angle())
            else:
                self.pa = np.zeros_like(self.coord1)

        else:
            self.pa = np.radians(self.pa)

        self.crval = kwargs.get('crval', np.array([np.median(self.coord1), np.median(self.coord2)]))

    def conversion(self, det_num=1):

        '''
        This function rotates the coordinates projected on the plane using the parallactic angle
        '''
        
        den = (np.sin(np.radians(self.coord2))*np.sin(np.radians(self.crval[1]))+\
               np.cos(np.radians(self.coord2))*np.cos(np.radians(self.crval[1]))*\
               np.cos(np.radians(self.coord1-self.crval[0])))

        x_proj = (np.cos(np.radians(self.coord2))*np.sin(np.radians(self.coord1-self.crval[0])))/den
        y_proj = (np.sin(np.radians(self.coord2))*np.cos(np.radians(self.crval[1]))-\
                  np.cos(np.radians(self.coord2))*np.sin(np.radians(self.crval[1]))*\
                  np.cos(np.radians(self.coord1-self.crval[0])))/den

        x_tel = -(x_proj*np.cos(self.pa)-y_proj*np.sin(self.pa))
        y_tel = (y_proj*np.cos(self.pa)+x_proj*np.sin(self.pa))

        if np.size(np.shape(self.coord1)) == 1:
            x_tel = np.tile(x_tel, (det_num, 1))
            y_tel = np.tile(y_tel, (det_num, 1))

        return np.degrees(x_tel), np.degrees(y_tel)

class offset:

    def __init__(self, proj):

        '''
        Class to handle pointing offsets. 
        '''

        self.proj = proj    


    def compute_offset(self, mp, pixel_coord, threshold=0.5, ref_point=None):

        '''
        Function to compute the offset of a particular detector. 
        Parameters:
        - mp: Map to be used for computing the offset 
        - pixel_coord: list with the pixel coordinate of the map 
        - threshold: threshold for the the determination of the centroid
        - ref_point: reference point to compute the offset. If none the crval of the 
                     map projection is used
        '''

        xc, yc = beam.centroid(mp, pixel_coord[0], pixel_coord[1], threshold=threshold)
        centroid_x, centroid_y = self.proj.all_pix2world(xc, yc, 1)

        if ref_point is None:
            ref_point = self.proj.wcs.crval

        
        return np.array([xc, yc]), np.array([centroid_x-ref_point[0], centroid_y-ref_point[1]])

    def apply_offset(self, coord1, coord2, offset_val):
        
        key = list(offset_val.keys())[0]
        offset_x = np.zeros(len(offset_val[key][0]))
        offset_y = np.zeros(len(offset_val[key][0]))

        for key in offset_val.keys():
            offset_x += offset_val[key][0]
            offset_y += offset_val[key][1]


        coord1 = (coord1.T-offset_x).T
        coord2 = (coord2.T-offset_y).T

        return coord1, coord2


        








        

        
        


