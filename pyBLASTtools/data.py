import pygetdata as gd
import numpy as np
import sys

from astropy.io import ascii

import pyBLASTtools.utils as utils

class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class InputError(Error):
    """Exception raised for errors in the input. """
    pass

class data:

    def __init__(self, path, idx_start, idx_end, field_list, mode='frames', ref_field=None):

        '''
        Class to handle dirfile: load, select a subsample or save data
        Parameters:
        - path: path of the dirfile
        - idx_start: first frame or sample of the subset.
        - idx_end: last frame or sample of the subset. For the last sample of the 
                   array, it is possible to use -1
        - field_list: a list of strings with the field to be loaded. If full all the 
                      fields in the dirfile are read
        - mode: can be frames or samples. If frames, idx_start(_end) are read as 
                as first and last frame of interest. If samples, the parameters idx_start(_end)
                refers to the a field given by the parameter ref_field. 
        - ref_field: reference field for the mode 'samples' to assign the idx_start(_end). 
                     If None, the reference field is considered the ctime
        '''

        self.d = gd.dirfile(path)

        if mode == 'frames':
            if idx_end == -1:
                idx_end = self.d.nframes

            first_frame = int(idx_start)
            num_frames = int(idx_end)-int(idx_start)
        else:
            if ref_field is None:
                self.ref_field = 'ctime_master_built'
            else:
                self.ref_field = ref_field

            if idx_end == -1:
                if field_list == 'full':                  
                    idx_end = self.d.array_len('ctime_master_built')
                else:
                    idx_end = self.d.array_len(self.ref_field)
        
        self.resample_completed = False
        self.data_values = {}

        if field_list == 'full':
            field_list = self.d.field_list()

        len_fields = np.array([])

        for i in field_list:
            if mode == 'frames':
                self.data_values[i] = self.d.getdata(i, first_frame=first_frame, num_frames=num_frames)
            else:
                first_sample = int(idx_start*self.d.spf(i)/self.d.spf(self.ref_field))
                num_samples = int((idx_end-idx_start)*self.d.spf(i)/self.d.spf(self.ref_field))

                self.data_values[i] = self.d.getdata(i, first_sample=first_sample, num_samples=num_samples)

            len_fields = np.append(len_fields, len(self.data_values[i]))

        if self.ref_field in field_list:
            self.ref_field_array = self.data_values[self.ref_field]
        else:
            self.ref_field_array = self.d.getdata(self.ref_field, first_sample=int(idx_start), \
                                                  num_samples=int(idx_end-idx_start))

        if np.all(np.diff(len_fields) == 0):
            self.resample_required = False
        else:
            self.resample_required = True

    def resample(self, field=None, interpolation_kind='linear'):

        '''
        Resample data based on a reference field. 
        Parameters:
        - field: field that needs to be resampled. It can be a string with the field name 
                 or a list with multiple fields. Default is None, which means that all 
                 the fields are resampled
        - interpolation_kind: the order of the interpolation for the resampling operation
        '''

        if field is None:
            for i in self.data_values.keys():
                
                if self.d.spf(i) == self.d.spf(self.ref_field):
                    pass
                else:
                    self.data_values[i] = utils.change_sampling_rate(self.ref_field_array,\
                                                                     self.data_values[i], \
                                                                     self.d.spf(self.ref_field), \
                                                                     self.d.spf(i), \
                                                                     interpolation_kind='linear')
            self.resample_completed = True
        else:
            if isinstance(field, str):
                self.data_values[field] = utils.change_sampling_rate(self.ref_field_array, \
                                                                     self.data_values[field], \
                                                                     self.d.spf(self.ref_field), \
                                                                     self.d.spf(field), \
                                                                     interpolation_kind='linear')

            else:
                for i in field:
                    if self.d.spf(i) == self.d.spf(self.ref_field):
                        pass
                    else:
                        self.data_values[i] = utils.change_sampling_rate(self.ref_field_array, \
                                                                         self.data_values[i], \
                                                                         self.d.spf(self.ref_field), \
                                                                         self.d.spf(i), \
                                                                         interpolation_kind='linear')

    def convert_to_array(self, interpolation_kind='linear'):

        '''
        Convert the dictionary data values to a numpy array. Numpy arrays needs to have 
        same dimensions, so a reference field is required if the resample function has not
        been run.
        Numpy array is built with the values from a single field on the same row. 
        '''

        try:
            if self.resample_completed is True or self.resample_required is False:
                keys = self.data_values.keys()
                array_val = np.zeros((len(keys), len(self.data_values[keys[0]])))
                count = 0
            else:
                if self.ref_field is not None:
                    keys = self.data_values.keys()
                    array_val = np.zeros((len(keys), len(self.ref_field_array)))
                    count = 0
                else:
                    raise InputError
        except InputError:
            print('The dictionary with the data has arrays with different lengths. \
                   Resampled is required to create an array')
            sys.exit(1)

        for i in self.data_values.keys():
            if self.d.spf(i) == self.d.spf(self.ref_field):
                array_val[count,:] = self.data_values[i]
            else:
                self.resample(field=i) 
                array_val[count,:] = self.self.data_values[i]

            count += 1

        return array_val

    def save(self, path, file_format):

        '''
        Function to save selected data in different formats. 
        It currently support saving only to csv or dirfile. Dirfile are currently created 
        with 1 sample per frame
        Parameters:
        - path: path of the file (including filename but not extension)
        - file_format: format of the file to be created
        '''
        
        format_list = ['csv', 'dirfile']

        try:
            if file_format in format_list:
                pass
            else:
                raise InputError

        except InputError:
            print('The file format choosen for saving the data is not implemented yet. \
                   Please choose a file format in', format_list)
            sys.exit(1)

        if file_format == 'csv':
            ascii.write(self.data_values, names=self.data_values.keys(), output=path+'.csv', format='csv')

        else:
            dd = gd.dirfile(path, gd.RDWR | gd.CREAT)

            for i in self.data_values.keys():
                if i in list(map(bytes.decode, dd.field_list())):
                    temp = dd.getdata(i)
                    if len(temp) == len(self.data_values[i]):
                        if np.all(np.diff(temp-self.data_values[i])==0):
                            pass
                else:
                    entry = gd.entry(gd.RAW_ENTRY, i, 0, (gd.FLOAT32))
                    dd.add(entry)

                dd.putdata(i, self.data_values[i], (gd.FLOAT32, 1))
                
            