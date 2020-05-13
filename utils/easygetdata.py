#!/usr/bin/python3

import sys
import numpy as np
import scipy as sp
import pygetdata as gd
import timeit

from typing import Union, List
from scipy import interpolate

class EasyGetData:
    SCALAR_ENTRIES = [gd.CONST_ENTRY, gd.CARRAY_ENTRY, gd.SARRAY_ENTRY, gd.STRING_ENTRY]

    def __init__(self, filename: str, access: str = "r"):
        """
        Primary constructor.

        Arguments:
        - filename: Path to the dirfile.
        - access:   Access string for handling dirfile. Follows system I/O rules as follows:
                    "rw" => Open for reading and writing. Create dirfile if doesn't exist.
                    "r"  => Open read-only. File must exist.
                    "w"  => Open new file for writing. Truncates existing file.
                    "r+" => Open for reading and writing. File must exist.
                    "w+" => Open for reading and writing. Truncates existing file.
        """
        flags = gd.RDONLY
        if access == "rw":
            flags = gd.RDWR | gd.CREAT
        elif access == "r":
            flags = gd.RDONLY
        elif access == "w":
            flags = gd.RDWR | gd.CREAT | gd.TRUNC
        elif access == "r+":
            flags = gd.RDWR
        elif access == "w+":
            flags = gd.RDWR | gd.TRUNC
        else:
            raise Exception("Invalid access string \"%s\"" % access)

        df = gd.dirfile(filename, flags=flags)

        self._df = df
        self._nframes = df.nframes
        self._field_names = [name.decode() for name in df.field_list() if self.is_vector(name)]


    def read_data(self, arange: List[int]=(0,-1), fields: List[str]=None, base_spf: int=None):
        """
        Read fields over an index range for a list of named fields.

        Arguments:
        - arange:   Tuple (start, end) to read frames [start, end).
                    Start and end allow negative indexing from EOF, where -1 is the last frame.
        - fields:   List of named fields in the dirfile to read.
        - base_spf: The effective samples-per-frame for each of the returned fields.
                    For a value of -1, all fields are upsampled to the max spf in the fields list.
                    If None, all fields retain their native spf.

        Returns:
        A dictionary of data blocks by field name.
        """
        # Sanitize arange
        start_frame, end_frame = arange
        if end_frame < 0:
            end_frame = max(0, end_frame + self._nframes + 1)
        if start_frame < 0:
            start_frame = max(0, start_frame + self._nframes + 1)
        num_frames = end_frame - start_frame

        # Sanitize fields
        if fields is None:
            fields = self._field_names
        fields = list(fields)

        # Sanitize spf
        if base_spf is not None and base_spf == -1:
            base_spf = max([self._df.spf(name) for name in fields])

        block = DataBlock()
        retdata = {}
        for name in fields:
            raw = self._df.getdata(
                    name,
                    first_frame=start_frame,
                    num_frames=num_frames,
                    first_sample=0,
                    num_samples=0,
                    return_type=gd.FLOAT64)

            if base_spf is None:
                processed = raw
            else:
                processed = self.resample(raw, base_spf * num_frames)
            retdata[name] = processed

        return retdata
    
    def resample(self, data, length: int, boxcar=True):
        """
        Resamples a 1D data vector to a new integer length.

        Arguments:
        - data:         The 1D data vector.
        - length:       The desired resampled length.
        - boxcar:       If True, applies boxcar filtering when downsampling
                        If False, applies simple decimation when downsampling
        """
        old_length = len(data)
        new_length = length
        if old_length == new_length:
            return data

        if new_length > old_length:
            # Upsample
            return self._upsample(data, new_length)
        else:
            # Downsample
            if old_length % new_length: 
                # Requires upsampling to nearest multiple first, then reducing
                data = self._upsample(data, int(np.ceil(old_length / new_length) * new_length))
                old_length = len(data)
            return self._downsample(data, int(old_length / new_length), boxcar=boxcar)

    def _downsample(self, data, factor: int, boxcar=True):
        """
        Downsamples a 1D data vector by an integer factor, either by decimation or boxcar filter.

        Arguments:
        - data:     The 1D data vector.
        - factor:   The factor by which data is downsampled (i.e. 2 => 1/2 the size).
                    len(data) must be divisible by factor.
        - boxcar:   If True, boxcar filters samples to get mean data point.
                    If False, simply decimates to every "factor" data points.

        Returns:
        The downsampled data vector.
        """
        length = len(data)
        if factor <= 1 or length == 0: return data
        if length % factor != 0:
            raise Exception("Data len %d is not divisible by %d" % (len(data), factor))

        if boxcar:
            # boxcar filter
            return data.reshape((-1, factor)).mean(axis=1)
        else:
            # decimation
            return data[::factor]

    def _upsample(self, data, length: int):
        """
        Upsamples a 1D data vector by an integer factor, either by repeated sample or interpolation.

        Arguments:
        - data:         The 1D data vector.
        - factor:       The factor by which data is upsampled (i.e. 2 => 2x the size).

        Returns:
        The upsampled data vector.
        """
        new_length = length
        old_length = len(data)
        if new_length <= old_length or old_length == 0: return data

        # linear interpolation
        input_x = np.linspace(0, old_length-1, old_length)
        output_x = np.linspace(0, old_length-1, new_length)
        return np.interp(output_x, input_x, data)

    def is_scalar(self, name):
        """
        Returns True if a named field is a scalar or string.

        Arguments:
        - name: Named field

        Returns:
        True or False depending on whether or not a field is a scalar or a string.
        """
        return self._df.entry(name).field_type in self.SCALAR_ENTRIES

    def is_vector(self, name):
        """
        Returns True if a named field is a vector (includes derived fields).

        Arguments:
        - name: Named field

        Returns:
        True or False depending on whether or not a field is a vector (includes derived fields).
        """
        return not self.is_scalar(name)

class DataBlock(dict):
    def __init__(self, ):
        self._raw_data = {}

    def add_field(data, name: str):
        self._raw_data[name] = data

def USAGE():
    print("EasyGetData v0.1\n\n"
            "Reads data from a dirfile and creates a new dirfile. "
            "Options for selecting data range by index, selecting specific fields, "
            "and upsampling/downsampling are available.\n\n"
            "Usage:\n"
            "easygetdata --infile=\"dirfile_in\" [--outfile=\"dirfile_out\"] [options]\n\n"
            "Options:\n"
            "--fields=field1[,field2[,..]]  Only output the given fields to the output dirfile\n"
            "--start=start_frame            Start reading input dirfile at given start frame\n"
            "--end=end_frame                End reading input dirfile at given end frame\n"
            "                               Start and end support reverse indexing from EOF\n"
            "                               (-1 is the last frame)\n"
            "                               Default start=0 and default end=-1 for whoel file\n"
            "--spf=sample_per_frame         The samples per frame to recast all data\n"
            "                               \"native\" -> each field retains its spf from infile\n"
            "                               \"max\" -> each field upsampled to max spf in fields\n"
            "                               All other values are interpreted as numeric\n"
            "                               Default spf is native\n"
            )
    exit(0)

def main():

    infile = None
    fields = None
    base_spf = None
    outfile = "output.DIR"
    start_frame = 0
    end_frame = -1
    for arg in sys.argv[1:]:
        optval = arg.split("=", 2)
        if optval[0] == "--fields":
            fields = [val.strip("\"") for val in optval[1].split(",")]
        elif optval[0] == "--infile":
            infile = optval[1].strip("\"")
        elif optval[0] == "--outfile":
            outfile = optval[1].strip("\"")
        elif optval[0] == "--start":
            start_frame = int(optval[1])
        elif optval[0] == "--end":
            end_frame = int(optval[1])
        elif optval[0] == "--spf":
            val = optval[1].strip("\"")
            if val == "native":
                base_spf = None
            elif val == "max":
                base_spf = -1
            else:
                base_spf = int(optval[1])
        else:
            print("Unrecognized option \"%s\"\n" % optval[0])
            USAGE()

    if infile is None:
        USAGE()

    df_in  = EasyGetData(infile, "r")
    df_out = EasyGetData(outfile, "w")
    a = df_in.read_data(arange=(start_frame, end_frame), fields=fields, base_spf=base_spf)

    for name, value in a.items():
        print(name)
        print(len(value))
        print(value)

if __name__ == "__main__":
    print(timeit.timeit(main, number=10) / 10)

        

