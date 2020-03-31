import numpy as np
import pygetdata as gd
import glob as gl
from itertools import compress
from scipy import interpolate
import sys

class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class InputError(Error):
    """Exception raised for errors in the input. """
    pass

### BLAST-TNG Detectors Parameters ###

clock_length = 1./(256*1e6) #Lenght of a single clock in s
clock_max = 2**32-1 # Numbers of clock counts before rolling over
fs  = 488.28125 # Sampling frequency for the roaches 

class timing():

    def __init__(self, path=None, master_path=None, roach_path=None):

        if path is not None:
            list_file = gl.glob(path+'/*') #Path of the folder with the different dirfiles
            
            self.roach_path = list(compress(list_file, ['roach' in s for s in list_file]))
            self.master_path = list_file['master' in list_file]
        
        if master_path is not None:
            self.master_path = master_path
        if roach_path is not None:
            if isinstance(roach_path, list):
                self.roach_path = roach_path
            else:
                self.roach_path = [roach_path]
    
    def ctime_master(self, write=False):

        '''
        Function to generate ctime for the master file
        '''

        if write:
            mode_dirfile = gd.RDWR
        else:
            mode_dirfile = gd.RDONLY

        self.d = gd.dirfile(self.master_path, mode_dirfile)
        self.time_master = (self.d.getdata('time')).astype(float)
        time_usec = (self.d.getdata('time_usec')).astype(float)

        self.time_master += time_usec/1.e6

        if write:

            self.write_ctime(spf=self.d.spf('time'))

        self.d.close()
    
    def ctime_roach(self, roach_number, kind, mode='average', write=False):

        '''
        Function to generate ctime for a given roach.
        roach_number: may be a str, int or float. List or numpy array is accepted too if time is 
                      required for multiple roaches
        kind: it is the method that is used to compute ctime for the roach. str for single entry 
              and list for multiple entries
              Possible methods:
              - Packet: A method that uses only packet count information
              - Clock: A method that uses only clock information
        mode: in case of multiple methods how the time information coming from the different
              methods will be combined
              Possible modes:
              - average
              - max
              - min
        write: choose to write the created ctime in the original dirfile
        '''

        kind_list = ['Packet', 'Clock']
        mode_list = ['average', 'max', 'min']

        try:
            if isinstance(roach_number, str) or isinstance(roach_number, int) or \
               isinstance(roach_number, float):
                self.time_roach = {}
                if isinstance(roach_number, str):
                    roach_number = roach_number.strip()

                roach_number = int(roach_number)
            
            elif isinstance(roach_number, list) or isinstance(roach_number, np.ndarray):               
                roach_number = np.array(roach_number).astype(int)
                self.time_roach = {}

                for i in range(len(roach_number)):
                    key = 'roach'+str(roach_number[i])
                    self.time_roach[key] = {}
            
            else:
                raise InputError
        
        except InputError:

            print('roach_number is not a str, int, float, list or numpy.array')
            sys.exit(1)

        try:
            if isinstance(kind, str):                
                kind = kind.strip()
                
                if kind in kind_list:
                    pass

                else:
                    raise InputError

            elif isinstance(kind, list):               
                kind = list(map(str.strip, kind))

                for i in kind:
                    if i in kind_list:
                        pass
                    else:
                        raise InputError
            
            else:                
                raise InputError
            
        except InputError:
            print('The method (kind) chosen to compute ctime_roach is not correct or is not a str or a list')
            sys.exit(1)

        try: 
            if isinstance(mode, str):
                if mode in mode_list:
                    pass
                else:
                    raise InputError

            else:
                raise InputError
        
        except InputError:
            print('The mode choosen for combininig the time methods calculation is not')
            print('between average, max or min. Or the mode is not a string')
            sys.exit(1)

        for i in range(np.size(roach_number)):

            
            if np.size(roach_number) == 1:
                if isinstance(roach_number, list):
                    roach_number_temp = roach_number[0]
                    key = 'roach'+str(roach_number[0])
                else:
                    roach_number_temp = roach_number
                    key = 'roach'+str(roach_number)
            else:
                roach_number_temp = roach_number[i]
                key = 'roach'+str(roach_number[i])

            if write:
                mode_dirfile = gd.RDWR
            else:
                mode_dirfile = gd.RDONLY

            self.d = gd.dirfile(self.roach_path[roach_number_temp-1], mode_dirfile)
            ctime_roach_name = 'ctime_roach'+str(int(roach_number_temp))
            self.ctime_roach_temp = (self.d.getdata(ctime_roach_name)).astype(np.float64)

            pps_roach_name = 'pps_count_roach'+str(int(roach_number_temp))
            self.pps_roach = (self.d.getdata(pps_roach_name)).astype(np.float64)

            for j in range(np.size(kind)):

                if np.size(kind) == 1:
                    if isinstance(kind, list):
                        kind_temp = kind[j]
                    else:    
                        kind_temp = kind
                else:
                    kind_temp = kind[j]

                if kind_temp.lower() == 'clock':
                    ctime_temp = self.clock_ctime_roach(roach_number_temp)

                elif kind_temp.lower() == 'packet':
                    ctime_temp = self.packet_ctime_roach(roach_number_temp)
                
                ctime_temp += 1570000000.
                ctime_temp += self.ctime_roach_temp*1e-2

                if j == 0:
                    ctime = ctime_temp
                else:
                    ctime = np.vstack((ctime_temp, ctime))

            del self.pps_roach
            del self.ctime_roach_temp

            if j != 0:
                print('ROACH ', key, 'completed')
                if mode == 'average':
                    self.time_roach[key] = np.average(ctime, axis=0)
                elif mode == 'max':
                    self.time_roach[key] = np.amax(ctime, axis=0)
                elif mode == 'min':
                    self.time_roach[key] = np.amin(ctime, axis=0)

            else:
                print('ROACH ', key, 'completed')
                self.time_roach[key] = ctime

            if write:

                self.write_ctime(roach_number_temp)
            
            self.d.close()

    def clock_ctime_roach(self, roach_number):

        '''
        Method to compute ctime for the roaches using the reference clock 
        '''

        clock_name = 'clock_count_roach'+str(int(roach_number))
        clock = (self.d.getdata(clock_name)).astype(np.float64)

        packet_name = 'packet_count_roach'+str(int(roach_number))
        packet = (self.d.getdata(packet_name)).astype(np.float64)	

        time = np.zeros(len(clock))

        #Check where clock is 0. That is a result of a missing packet

        idx_zero, = np.where(clock == 0)

        #Assign values at each ctime change

        idx_non_zero, = np.where(self.ctime_roach_temp != 0)
        idx_change_temp, = np.where((np.diff(self.ctime_roach_temp[idx_non_zero]) > 0))
        idx_change_temp += 1
        idx_change = idx_non_zero[idx_change_temp] 	

        idx_change = np.append(0, np.append(idx_change, len(clock)-1))

        for j in range(len(idx_change)-1):

            idx_temp, = np.where(self.pps_roach[idx_change[j]:idx_change[j+1]] == self.pps_roach[idx_change[j]])

            delta_packet0 = packet[idx_change[j]+idx_temp[-1]]-packet[idx_change[j]+idx_temp[0]]

            t0_pps_max = (self.pps_roach[idx_change[j]]+1)-delta_packet0/fs
            t0_pps_min = (self.pps_roach[idx_change[j]]+1)-(delta_packet0+1.)/fs
            time[idx_change[j]]= (t0_pps_max+t0_pps_min)/2.

        #Assign time values at roll over

        idx_roll, = np.where(np.diff(clock)<0)
        idx_roll += 1
        
        #Indices at the roll over (and not when packets are lost) 
        idx_roll_real, = np.where((clock[idx_roll] > 0) & \
                                  ((self.ctime_roach_temp[idx_roll]-self.ctime_roach_temp[idx_roll-1])==0)) 
        idx_roll_zero, = np.where(clock[idx_roll] == 0)	#Indices of the first lost packets

        time[idx_roll[idx_roll_real]] = (clock[idx_roll[idx_roll_real]] + clock_max+1-\
                                         clock[idx_roll[idx_roll_real]-1])*clock_length

        #Assign time values for sequential samples 

        mask = np.ones(len(clock), dtype = bool)
        mask[idx_roll] = False
        mask[idx_zero] = False
        mask[idx_change] = False
        idx_seq = np.arange(len(clock))[mask]
        idx_seq = idx_seq[1:]	

        idx_seq_zero, = np.where(clock[idx_seq-1] == 0) #Indices where the previous element is zero and not at any roll over
        idx_seq_nonzero, = np.where(clock[idx_seq-1] != 0) #Indices where the previous element is different from zero and not at any roll over

        time[idx_seq[idx_seq_nonzero]] = ((clock[idx_seq[idx_seq_nonzero]]-clock[idx_seq[idx_seq_nonzero]-1])*\
                                          clock_length)

        # Compute time after a period of zeros 
        if idx_seq_zero.size:
            
            #Loop through multiple gaps
            for ii in range(len(idx_seq_zero)):

                diff = idx_seq[idx_seq_zero[ii]]-idx_roll[idx_roll_zero[ii]]-1
                
                #If the period of zeros is smaller than the maximum number of clock cycles
                if diff/fs < clock_max*clock_length:

                    if clock[idx_seq[idx_seq_zero[ii]]]> clock[idx_roll[idx_roll_zero[ii]]-1]:
                        fact = 0
                    else:
                        fact = clock_max+1
                    
                    time[idx_seq[idx_seq_zero[ii]]] = (clock[idx_seq[idx_seq_zero[ii]]]+fact-\
                                                       clock[idx_roll[idx_roll_zero[ii]]-1])*clock_length
                else:
                    if clock[idx_seq[idx_seq_zero[ii]]]<clock[idx_roll[idx_roll_zero[ii]]-1]:
                        fact = np.floor(diff/fs/clock_max/clock_length)
                    else:
                        fact = np.floor(diff/fs/clock_max/clock_length)-1

                    time[idx_seq[idx_seq_zero[ii]]] = (clock[idx_seq[idx_seq_zero[ii]]]+(fact+1)*(clock_max+1)-\
                                                       clock[idx_roll[idx_roll_zero[ii]]-1])*clock_length
                                               


        # Create final array with correction to include possible clock jumps

        for j in range(len(idx_change)-1):

            time[idx_change[j]:idx_change[j+1]] = np.nancumsum(time[idx_change[j]:idx_change[j+1]])
            idx_jump, = np.where(np.abs(np.diff(time[idx_change[j]:idx_change[j+1]])-1./fs)>0.00001)
            idx_jump += 1
            for k in range(len(idx_jump)):
                idx_temp = idx_change[j]+idx_jump[k]
                if packet[idx_temp]-packet[idx_temp-1] == 1:
                    time[idx_temp:idx_change[j+1]] += -(time[idx_temp]-time[idx_temp-1] -1./fs)
                
        time[idx_zero] = np.nan

        del clock
        del packet

        return time

    def packet_ctime_roach(self, roach_number):

        '''
        Method to compute ctime for the roaches using packet counts
        '''

        packet_name = 'packet_count_roach'+str(int(roach_number))
        packet = (self.d.getdata(packet_name)).astype(np.float64)	
        
        time = np.zeros(len(packet))	

        idx_zero, = np.where(packet == 0)

        idx_non_zero, = np.where(self.ctime_roach_temp != 0)
        idx_change_temp, = np.where((np.diff(self.ctime_roach_temp[idx_non_zero]) > 0))
        idx_change_temp += 1
        idx_change = idx_non_zero[idx_change_temp] 
        
        idx_change = np.append(0, np.append(idx_change, len(packet)-1))

        for j in range(len(idx_change)-1):
            
            packet_temp = packet[idx_change[j]:idx_change[j+1]]

            idx_temp, = np.where(self.pps_roach[idx_change[j]:idx_change[j+1]] == self.pps_roach[idx_change[j]])

            delta_packet0 = packet_temp[idx_temp[-1]]-packet_temp[idx_temp[0]]

            t0_pps_max = (self.pps_roach[idx_change[j]]+1)-(delta_packet0)/fs
            t0_pps_min = (self.pps_roach[idx_change[j]]+1)-(delta_packet0+1)/fs
            time[idx_change[j]]= (t0_pps_max+t0_pps_min)/2.

            array = np.arange(0, len(packet[idx_change[j]:idx_change[j+1]]), 1.)
            
            time[idx_change[j]:idx_change[j+1]] += array/fs
            time[idx_change[j]+1:idx_change[j+1]] += time[idx_change[j]]

            idx_zero_temp, = np.where(self.ctime_roach_temp[idx_change[j]:idx_change[j+1]] != 0)

            if idx_zero_temp.size:
                idx_jump, = np.where(np.diff(packet_temp[idx_zero_temp]) != 1)

                for i in range(len(idx_jump)):
                    idx_jump_temp = idx_zero_temp[idx_jump[i]+1]
                    idx_start = idx_jump_temp+idx_change[j]

                    if packet_temp[idx_jump_temp-1] != 0:
                        time[idx_start:idx_change[j+1]] += (packet[idx_start]-packet[idx_start-1]-1)*1/fs

                    else:
                        idx_jump_prev = idx_zero_temp[idx_jump[i-1]+1]
                        idx_start_prev = idx_jump_prev+idx_change[j]

                        idx_non_zero_prev, = np.where(packet[idx_start_prev:idx_start]!=0)
                        idx_zero_prev, = np.where(packet[idx_start_prev:idx_start]==0)
                        
                        packet_temp_non_zero = packet[idx_start_prev+idx_non_zero_prev[-1]]
                        delta_packet = packet_temp_non_zero+len(idx_zero_prev)

                        time[idx_start:idx_change[j+1]] += (packet_temp[idx_jump_temp]-delta_packet-1)*1/fs

        del packet

        time[idx_zero] = np.nan

        return time

    def write_ctime(self, roach_number = None, spf=1):

        if roach_number is not None:
            ctime_name = 'ctime_built_roach'+str(int(roach_number))
            key = 'roach'+str(int(roach_number))
            val = self.time_roach[key]
        else:
            ctime_name = 'ctime_master_built'
            val = self.time_master

        if ctime_name in list(map(bytes.decode, self.d.field_list())):
            pass
        else:
            ctime_entry = gd.entry(gd.RAW_ENTRY, ctime_name, 0, (gd.FLOAT64, spf))
            self.d.add(ctime_entry)
        self.d.putdata(ctime_name, val , gd.FLOAT64)

class dirfile_interp():

    '''
    Class to interpolate the data between two different sampling data sources.
    This class is considering the the ctime for the master and the roach has been built.
    '''

    def __init__(self, loading_method, path_master=None, path_roach=None, roach_num=1, idx_start=0, idx_end=-1, \
                 time_start=None, time_end=None, time_master=None, time_roach=None, offset=0.):

        '''
        Input parameters for interpolating the data:
        loading_method: a string between:
                        - idx: Using two indices on the master file to select the data
                        - idx_roach: Using two indices of the roach file to select data
                        - time_val: Using two time values on the master file to select at the data
                        - time_array: using two time arrays, one from master and one from the roach as 
                                      reference
        path_master: path of the master file
        path_roach: path of the roach file
        roach_num: which roach is going to be analyzed
        idx_start: Starting index of the data that need to be analyzed. The index is from the 
                   master file if the loading method is 'idx' and is from the roach file if the 
                   loading method is 'idx_roach'
        idx_end: Ending index of the data that need to be analyzed. The index is from the 
                 master file if the loading method is 'idx' and is from the roach file if the 
                 loading method is 'idx_roach'
        time_start: Starting time of the data that need to be analyzed. The time is from the 
                    master file. The array needs to be already sliced 
        time_end: Ending time of the data that need to be analyzed. The time is from the 
                  master file. The array needs to be already sliced
        time_master: Array with the time data from master file
        time_roach: Array with the time data from one of the roach file
        offset: time offset in seconds of the roach time array with respect to the master time.
                This is defined as the time to be added (or subctrated if negative) to the master time. 
                This offset is not applied in case it is used the 'time_array' option for loading
                the time arrays.
        '''

        loading_method_list = ['idx', 'idx_roach', 'time_val', 'time_array']

        try:
            if loading_method.strip().lower() in loading_method_list:
                pass
            else:
                raise InputError
        except InputError:
            print('The loading method choosen is not correct. Choose between: idx, time_val, time_array')
            sys.exit(1)

        if loading_method.strip().lower() == 'time_array':

            self.time_master = time_master
            self.time_roach = time_roach

        else:

            self.d_master = gd.dirfile(path_master)
            self.d_roach = gd.dirfile(path_roach)

            self.roach_num = roach_num
            roach_time_str = 'ctime_built_roach'+str(int(self.roach_num))

            if loading_method.strip().lower() == 'idx_roach':
                self.time_roach = self.d_roach.getdata(roach_time_str)

                self.idx_start_roach = idx_start
                self.idx_end_roach = idx_end

                self.time_roach = self.time_roach[self.idx_start_roach:self.idx_end_roach]

                self.time_master = self.d_master.getdata('ctime_master_built')

                self.idx_start_master = np.nanargmin(np.abs(self.time_master-self.time_roach[0]+offset))
                self.idx_end_master = np.nanargmin(np.abs(self.time_master-self.time_roach[-1]+offset))
                
                self.time_master = self.time_master[self.idx_start_master:self.idx_end_master]

            else:
                if loading_method.strip().lower() == 'idx':

                    self.time_master = self.d_master.getdata('ctime_master_built')

                    self.idx_start_master = idx_start
                    self.idx_end_master = idx_end

                    self.time_master = self.time_master[self.idx_start_master:self.idx_end_master]

                elif loading_method.strip().lower() == 'time_val':

                    self.time_master = self.d_master.getdata('ctime_master_built')

                    self.idx_start_master = np.nanargmin(np.abs(self.time_master-time_start))
                    self.idx_end_master = np.nanargmin(np.abs(self.time_master-time_end))

                    self.time_master = self.time_master[self.idx_start_master:self.idx_end_master]

                self.time_roach = self.d_roach.getdata(roach_time_str)

                self.idx_start_roach = np.nanargmin(np.abs(self.time_roach-self.time_master[0]-offset))
                self.idx_end_roach = np.nanargmin(np.abs(self.time_roach-self.time_master[-1]-offset))

                self.time_roach = self.time_roach[self.idx_start_roach:self.idx_end_roach]
                
    def interp(self, field_master, field_roach, direction='mtr', interpolation_type='linear'):

        '''
        Method for loading the data from the dirfile and then calling the interpolation method
        List of variables:
        - field_master: a string with the field coming from master
        - field_roach: a string with the field 
        - direction: the direction of the interpolation
                     - mtr: interpolating master to roach
                     - rtm: interpolating roach to master 
        - interpolation_type: which kind of interpolation to be used. Standard scipy.interp1d values
        '''

        #On master every field has a different number of sample per frame
        #The index on the time needs to be shifted to take into consideration the different spf

        spf_ctime = self.d_master.spf('ctime_master_built')
        spf_field = self.d_master.spf(field_master)

        if spf_ctime != spf_field:
            field_master_array = self.master_to_100(field_master, spf_field, spf_ctime)
        
        else:
            field_master_array = self.d_master.getdata(field_master)
            field_master_array = field_master_array[self.idx_start_master:self.idx_end_master]
 
        field_roach_array = self.d_roach.getdata(field_roach, first_frame=self.idx_start_roach, \
                                                 num_frames=int(self.idx_end_roach-self.idx_start_roach))

        field_master, field_roach = self.interpolate(field_master_array, field_roach_array, \
                                                     direction=direction, interpolation_type=interpolation_type)

        return field_master, field_roach

    def master_to_100(self, field_master, spf_field=None, spf_ctime=None):

        '''
        Function to convert master fields from a certain sampling frequency to 100 Hz.
        This function return only the sliced array 
        '''

        if spf_field is None:
            spf_field = self.d_master.spf(field_master)

        if spf_ctime is None:
            spf_ctime = self.d_master.spf('ctime_master_built')

        idx_start_master_temp = int(np.floor(self.idx_start_master*spf_field/spf_ctime))
        idx_end_master_temp = int(np.floor(self.idx_end_master*spf_field/spf_ctime))
        
        field_master_array = self.d_master.getdata(field_master)
        field_master_array = field_master_array[idx_start_master_temp:idx_end_master_temp]

        x_axis = np.arange(len(field_master_array))/(spf_field/spf_ctime)

        f = interpolate.interp1d(x_axis, field_master_array, kind='linear', fill_value='extrapolate')
        field_master_array = f(np.arange(len(self.time_master)))

        assert len(field_master_array) == len(self.time_master)

        return field_master_array

    def interpolate(self, master_array, roach_array, direction='mtr', interpolation_type='linear'):

        '''
        Method for interpolating the data.
        List of variables:
        - master_array: array with the data coming from master
        - roach_array: array with the data coming from roach
        - direction: the direction of the interpolation
                     - mtr: interpolating master to roach
                     - rtm: interpolating roach to master 
        - interpolation_type: which kind of interpolation to be used. Standard scipy.interp1d values
        '''

        try: 
            if direction in ['mtr', 'rtm']:
                pass
            else:
                raise InputError

        except InputError:
            print('The direction choosen for the interpolation is not correct. Choose between mtr and rtm')
            sys.exit(1)

        if direction == 'rtm':

            f = interpolate.interp1d(self.time_roach, roach_array, kind=interpolation_type, \
                                     fill_value='extrapolate')

            return master_array, f(self.time_master)
        
        elif direction == 'mtr':

            f = interpolate.interp1d(self.time_master, master_array, kind=interpolation_type, \
                                     fill_value='extrapolate')
            
            return f(self.time_roach), roach_array

        

        

        



