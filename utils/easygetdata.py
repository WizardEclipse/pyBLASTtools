#!/usr/bin/python3
import sys
import numpy as np
import pygetdata as gd

from collections.abc import Mapping
from typing import List

"""
Basic reverse lookup for GetData types based on NumPy dtype.
"""
GDTYPE_LOOKUP = {
        np.dtype("int8"):        gd.INT8,
        np.dtype("int16"):       gd.INT16,
        np.dtype("int32"):       gd.INT32,
        np.dtype("int64"):       gd.INT64,
        np.dtype("uint8"):       gd.UINT8,
        np.dtype("uint16"):      gd.UINT16,
        np.dtype("uint32"):      gd.UINT32,
        np.dtype("uint64"):      gd.UINT64,
        np.dtype("float"):       gd.FLOAT,
        np.dtype("float32"):     gd.FLOAT32,
        np.dtype("float64"):     gd.FLOAT64,
        np.dtype("complex"):     gd.COMPLEX,
        np.dtype("complex64"):   gd.COMPLEX64,
        np.dtype("complex128"):  gd.COMPLEX128}

class EasyGetData:
    """
    A generic class for Python-izing reading and writing dirfile using pygetdata.
    """
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
        self.nframes = df.nframes
        self.field_names = [
                name.decode() 
                for name in df.field_list() 
                if self.is_vector(name) and name != b"INDEX"]


    def read_data(self, arange: List[int]=(0,-1), fields: List[str]=None, spf: int=None):
        """
        Read fields over an index range for a list of named fields.

        Arguments:
        - arange:   Tuple (start, end) to read frames [start, end).
                    Start and end allow negative indexing from EOF, where -1 is the last frame.
        - fields:   List of named fields in the dirfile to read.
        - spf:      The effective samples-per-frame for each of the returned fields.
                    For values < 0, all fields are upsampled to the max spf in the fields list.
                    If None, no resampling is done.

        Returns:
        A DataBlock that manages data manipulation (resampling) of the raw data.
        """
        # Sanitize arange
        start_frame, end_frame = arange
        if end_frame < 0:
            end_frame = max(0, end_frame + self.nframes + 1)
        if start_frame < 0:
            start_frame = max(0, start_frame + self.nframes + 1)
        num_frames = end_frame - start_frame

        # Sanitize fields
        if fields is None:
            fields = self.field_names
        fields = list(fields)

        # Sanitize spf
        if spf is not None and spf == -1:
            spf = max([self._df.spf(name) for name in fields])

        data = {}
        for name in fields:
            # getdata from the dirfile
            raw = self._df.getdata(
                    name, 
                    first_frame=start_frame, 
                    num_frames=num_frames, 
                    first_sample=0, 
                    num_samples=0)
            raw_spf = int(len(raw) / num_frames)
            data[name] = RawDataField(raw, raw_spf)
        # Return the data in the form of a DataBlock, unless dealing with raw data
        return DataBlock(data=data, nframes=num_frames, spf=spf)

    def write_data(self, data: Mapping, arange: List[int] = (0, -1)):
        """
        Writes a data block to dirfile over an index range.

        Arguments:
          data:     The data block to be written.
        - arange:   Tuple (start, end) to read frames [start, end).
        """
        # Sanitize arange
        start_frame, end_frame = arange
        if end_frame < 0:
            end_frame = max(0, end_frame + data.nframes + 1)
        if start_frame < 0:
            start_frame = max(0, start_frame + data.nframes + 1)
        num_frames = end_frame - start_frame

        for name, value in data.items():
            if data.spf is None:
                spf = value.spf 
            else:
                spf = data.spf
            v = np.array(value[(start_frame*spf):(end_frame*spf)])
            # Create the entry if not already present in the dirfile
            if name not in self._df.field_list():
                t = GDTYPE_LOOKUP[value.dtype]
                entry = gd.entry(gd.RAW_ENTRY, name, 0, (t ,spf)) 
                self._df.add(entry)

            # putdata into the dirfile
            print("%20s => %d frames (%d spf) starting at frame %d" % 
                    (name, num_frames, spf, start_frame))
            self._df.putdata(name, v, first_frame=start_frame)

        self._df.flush()

        return num_frames


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

class RawDataField(np.ndarray):
    def __new__(cls, input_array, spf: int):
        obj = np.asarray(input_array).view(cls)
        obj.nsamples = len(input_array)
        obj.nframes = int(obj.nsamples / spf)
        obj.spf = spf
        return obj

    def __array_finalize__(self, obj):
        if obj is None: 
            return
        self.nsamples = getattr(obj, 'nsamples', None)
        self.nframes = getattr(obj, 'nframes', None)
        self.spf = getattr(obj, 'spf', None)

class DataBlock(Mapping):
    """
    Subclass of a typical mapping, or dictionary that wraps a structured record numpy array.

    Supports dict-like lookup by field name and array-like lookup by index.

    For index lookup, data is returned per frame (i.e. 1 frame == 1 index), so for data with N
    samples per frame (spf), N samples are returned per field per frame.
    """
    def __init__(self, data, nframes=0, spf=None):
        """
        DataBlock initializer. 

        Takes a dict-like data object and converts it to a structure record numpy array, where the
        format is based on field name (i.e. the columns are data fields).

        Arguments:
        - data:     A dict-like Mapping containing the generator data. Valid types are:
                        dict        => The base generator for data as {name: data}.
                        DataBlock   => Creates a copy that is based on the same generator.
        - nframes:  The maximumn number of frames/indices implied by the input data object.
        - spf:      The desired number of samples-per-frame.
        """

        self.nframes = nframes
        self.spf = spf

        self._cube = None
        self._cube_spf = None 

        if isinstance(data, dict):
            # Set the input dict as the generator
            self._generator = data
        elif isinstance(data, DataBlock):
            # Copy attributes from input DataBlock
            self._generator = data._generator 
        self._regenerate()

    @property
    def data(self):
        """
        Returns the naked 2D array representing the data.
        Only has meaning for non-native DataBlock with homogenous spf.
        """
        return self._cube.view().reshape(self._cube.shape + (-1,))
    
    @property
    def _active_data(self):
        """
        Returns the active data object depending on whether or not a data cube has been generated.
        """
        if self.spf is None:
            return self._generator
        return self._cube

    def _names(self):
        """
        Returns names.
        """
        if self.spf is None:
            return self._generator
        return self._cube.dtype.names

    def _regenerate(self):
        """
        An internal function that updates the data cube based on spf attributes. 
        If the DataBlock had changed spf, the data cube is regenerated. 
        Otherwise no action is taken.
        """
        if self.spf is None:
            return
        if self._cube_spf != self.spf or self._cube_spf is None:
            # Compute number of samples per data field
            nsamples = self.nframes * self.spf
            # Generate the data cube
            dtype = [(name, "f8") for name in self._generator]
            self._cube = np.zeros(nsamples, dtype=dtype)
            for name, raw in self._generator.items():
                self._cube[name] = self.resample(raw, nsamples)
            # Set the cube state to the achieved goal
            self._cube_spf = self.spf

    def __getitem__(self, key):
        """
        Overload of the lookup by dict key.

        Arguments:
        - key:      The string field name, index integer, or slice to lookup data.

        Returns:
        For string field name, returns the data array.

        For index integer, returns a structured array of all the fields at a given frame/index.
        For spf > 1, spf samples per field are provided.

        For slice, returns a structure array of all the fields over the given frame/index range.
        For spf > 1, spf samples per field per index are provided.
        """
        # Regenerate data cube, if needed.
        self._regenerate()
        data = self._active_data

        # Getting data from index
        if isinstance(key, int):
            key = slice(key, key+1)
        if isinstance(key, slice):
            if self.spf is None:
                raise Exception("Cannot index inhomogenous data with spf=None")
            if key.start is not None: start = key.start * self.spf
            else: start = None
            if key.stop is not None: stop = key.stop * self.spf
            else: stop = None
            if key.step is not None: step = key.step * self.spf
            else: step = None
            key = slice(start, stop, step)

        # Getting data by field name
        return data[key]

    def __iter__(self):
        """
        Iterate by column of field data.
        """
        return iter(self._names())

    def __len__(self):
        """
        Returns the number field data columns.
        """
        return len(self._names())
    
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
        if new_length == 0 or old_length == 0:
            return np.array([])

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
    sys.exit(0)

def main():

    infile = None
    fields = None
    spf = None
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
                spf = None
            elif val == "max":
                spf = -1
            else:
                spf = int(optval[1])
        else:
            print("Unrecognized option \"%s\"\n" % optval[0])
            USAGE()

    if infile is None:
        USAGE()

    df_in  = EasyGetData(infile,  "r")
    df_out = EasyGetData(outfile, "w")

    if fields is None:
        fields = df_in.field_names

    for field in fields:
        arange=(start_frame, end_frame)
        data = df_in.read_data(arange=arange, fields=[field], spf=spf)
        df_out.write_data(data=data, arange=arange)

if __name__ == "__main__":
    main()

        

