import numpy as np
import pygetdata as gd
import glob as gl
from itertools import compress
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

    def __init__(self, path):

        list_file = gl.glob(path+'/*') #Path of the folder with the different dirfiles
        
        self.roach_path = list(compress(list_file, ['roach' in s for s in list_file]))
        self.master_path = list_file['master' in list_file]
    
    def ctime_master(self):

        '''
        Function to generate ctime for the master file
        '''

        d = gd.dirfile(self.master_path)
        self.time_master = (d.getdata('time')).astype(float)
        time_usec = (d.getdata('time_usec')).astype(float)

        self.time_master += time_usec/1.e6

    
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
                    kind_temp = kind
                else:
                    kind_temp = kind[j]

                if kind_temp.lower() == 'clock':
                    ctime_temp = self.clock_ctime_roach(roach_number_temp)

                elif kind_temp.lower() == 'packet':
                    ctime_temp = self.packet_ctime_roach(roach_number_temp)

                elif kind_temp.lower() == 'ppsclock':
                    ctime_temp = self.ppsclock_ctime_roach(roach_number_temp)
                
                elif kind_temp.lower() == 'ppspacket':
                    ctime_temp = self.ppspacket_ctime_roach(roach_number_temp)
                
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

    def write_ctime(self, roach_number):

        ctime_name = 'ctime_built_roach'+str(int(roach_number))
        key = 'roach'+str(int(roach_number))

        if ctime_name in list(map(bytes.decode, self.d.field_list())):
            pass
        else:
            ctime_entry = gd.entry(gd.RAW_ENTRY, ctime_name, 0, (gd.FLOAT64, 1))
            self.d.add(ctime_entry)
        self.d.putdata(ctime_name, self.time_roach[key], gd.FLOAT64)



