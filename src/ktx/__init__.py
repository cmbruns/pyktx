"""
https://www.khronos.org/opengles/sdk/tools/KTX/file_format_spec/

Copyright (c) 2016 Christopher M. Bruns

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import collections
import io
import struct
import sys

from OpenGL import GL

from .version import __version__
from ktx.util import create_mipmaps, interleave_channel_arrays, mipmap_dimension

class Ktx(object):
    
    def __init__(self):
        self.header = KtxHeader()
        self.image_data = KtxImageData()
    
    def asarray(self, mipmap_level=0):
        import numpy
        # Extract image shape
        h = self.header
        full_shape = list()
        if h.pixel_depth > 1:
            full_shape.append(h.pixel_depth)
        if h.pixel_height > 1:
            full_shape.append(h.pixel_height)
        full_shape.append(h.pixel_width)
        # Adjust shape for mipmap level
        mipmap_shape = [mipmap_dimension(mipmap_level, d) for d in full_shape]
        # Add channel dimension, if multichannel
        if h.gl_base_internal_format == GL.GL_RG:
            mipmap_shape.append(2)
        elif h.gl_base_internal_format == GL.GL_RGB:
            mipmap_shape.append(3)
        elif h.gl_base_internal_format == GL.GL_RGBA:
            mipmap_shape.append(4)
        else:
            assert h.gl_base_internal_format == GL.GL_RED
        shape = tuple(mipmap_shape)
        if h.gl_type == GL.GL_UNSIGNED_SHORT:
            dtype = numpy.uint16
        elif h.gl_type == GL.GL_UNSIGNED_BYTE:
            dtype = numpy.uint8
        else:
            assert False # TODO: more data types...
        buffer = self.image_data.mipmaps[mipmap_level] # array of bytes
        result = numpy.ndarray(buffer=buffer, dtype=dtype, shape=shape)
        return result
    
    @staticmethod
    def from_ndarray(array, multichannel=None, mipmap_filter='arthur'):
        """
        Creates a Ktx object from an array like that returned by tifffile.TiffFile.asarray()
        """
        # Does this array have multiple color channels?
        if multichannel is None:
            # Guess, in case multichannel is not specified
            multichannel = len(array.shape) > 1 and array.shape[-1] < 5
        if multichannel:
            # yes, multichannel
            spatial_shape = array.shape[0:-1]
            channel_count = array.shape[-1]
        else:
            spatial_shape = array.shape
            channel_count = 1
        # print (spatial_shape, channel_count)
        ktx = Ktx() 
        dt = array.dtype
        kh = ktx.header
        if dt.byteorder == '<':
            kh.little_endian = True
        elif dt.byteorder == '=':
            kh.little_endian = sys.byteorder == 'little'
        else:
            raise # TODO
        # print (dt.byteorder)
        # print (kh.little_endian)
        if dt.kind == 'u':
            if dt.itemsize == 2:
                kh.gl_type = GL.GL_UNSIGNED_SHORT
            elif dt.itemsize == 1:
                kh.gl_type = GL.GL_UNSIGNED_BYTE
            else:
                raise # TODO:
        else:
            raise # TODO:
        kh.gl_type_size = dt.itemsize
        #
        if channel_count == 1:
            kh.gl_format = kh.gl_base_internal_format = GL.GL_RED
        elif channel_count == 2:
            kh.gl_format = kh.gl_base_internal_format = GL.GL_RG
        elif channel_count == 3:
            kh.gl_format = kh.gl_base_internal_format = GL.GL_RGB
        elif channel_count == 4:
            kh.gl_format = kh.gl_base_internal_format = GL.GL_RGBA
        else:
            raise # TODO
        #
        # TODO: - lots of cases need to be enumerate here...
        if kh.gl_base_internal_format == GL.GL_RG:
            if kh.gl_type == GL.GL_UNSIGNED_SHORT:
                kh.gl_internal_format = GL.GL_RG16UI
            else:
                raise
        elif kh.gl_base_internal_format == GL.GL_RGB:
            if kh.gl_type == GL.GL_UNSIGNED_SHORT:
                kh.gl_internal_format = GL.GL_RGB16UI
            else:
                raise
        else:
            raise # TODO
        #
        kh.pixel_width = spatial_shape[2]
        kh.pixel_height = spatial_shape[1]
        kh.pixel_depth = spatial_shape[0]
        kh.number_of_array_elements = 0
        kh.number_of_faces = 0
        
        # 
        ktx.image_data.mipmaps.clear()
        if mipmap_filter is not None:
            channel_mipmaps = list()
            for c in range(channel_count):
                # Prepare to split exactly one channel from array
                channel_key = [slice(None),] * len(spatial_shape)
                if multichannel:
                    channel_key.append(c)
                channel = array[channel_key]
                mipmaps = create_mipmaps(channel, filter_=mipmap_filter)
                channel_mipmaps.append(mipmaps)
            # Recombine channels, one mipmap at a time
            kh.number_of_mipmap_levels = len(channel_mipmaps[0])
            for m in range(kh.number_of_mipmap_levels):
                channels = [a[m] for a in channel_mipmaps]
                combined = interleave_channel_arrays(channels)
                ktx.image_data.mipmaps.append(combined.tostring())
        else:
            kh.number_of_mipmap_levels = 1
            ktx.image_data.mipmaps.append(array.tostring())
        return ktx
    
    def read_filename(self, file_name):
        with io.open(file_name, 'rb') as fh:
            self.read_stream(fh)
    
    def read_stream(self, file):
        self.header.read_stream(file)
        self.image_data.read_stream(file, self.header)
        
    def write_stream(self, file):
        self.header.write_stream(file)
        self.image_data.write_stream(file, self.header)
        

class KtxParseError(Exception):
    pass


class KtxHeader(object):
    
    def __init__(self):
        self.key_value_metadata = collections.OrderedDict()
    
    # Expose key_value_metadata as top level dictionary
    def __len__(self):
        return len(self.key_value_metadata)
    
    def __getitem__(self, key):
        return self.key_value_metadata[str(key).encode()]
    
    def __setitem__(self, key, value):
        self.key_value_metadata[str(key).encode()] = str(value).encode()
        
    def __delitem__(self, key):
        del self.key_value_metadata[str(key).encode()]
        
    def __iter__(self):
        return self.key_value_metadata.__iter__()
    
    def __contains__(self, item):
        return str(item).encode() in self.key_value_metadata
    
    def read_stream(self, file):
        # First load and check the binary file identifier for KTX files
        identifier = file.read(12)
        """
        The rationale behind the choice values in the identifier is based on 
        the rationale for the identifier in the PNG specification. This 
        identifier both identifies the file as a KTX file and provides for 
        immediate detection of common file-transfer problems. 
         * Byte [0] is chosen as a non-ASCII value to reduce the probability 
           that a text file may be misrecognized as a KTX file.
         * Byte [0] also catches bad file transfers that clear bit 7.
         * Bytes [1..6] identify the format, and are the ascii values for the 
           string "KTX 11".
         * Byte [7] is for aesthetic balance with byte 1 (they are a matching 
           pair of double-angle quotation marks).
         * Bytes [8..9] form a CR-LF sequence which catches bad file 
           transfers that alter newline sequences.
         * Byte [10] is a control-Z character, which stops file display 
           under MS-DOS, and further reduces the chance that a text file will be falsely recognised.
         * Byte [11] is a final line feed, which checks for the inverse of 
           the CR-LF translation problem.
        """
        expected_identifier = b'\xabKTX 11\xbb\r\n\x1a\n'
        if identifier != expected_identifier:
            raise KtxParseError('KTX binary file identifier not found. Found %s, expected %s' % (identifier, expected_identifier))
        # Endianness
        endian = file.read(4)
        """
        endianness contains the number 0x04030201 written as a 32 bit integer. 
        If the file is little endian then this is represented as the bytes 
        0x01 0x02 0x03 0x04. If the file is big endian then this is 
        represented as the bytes 0x04 0x03 0x02 0x01. When reading 
        endianness as a 32 bit integer produces the value 0x04030201 then 
        the endianness of the file matches the the endianness of the 
        program that is reading the file and no conversion is necessary. 
        When reading endianness as a 32 bit integer produces the value 
        0x01020304 then the endianness of the file is opposite the 
        endianness of the program that is reading the file, and in that 
        case the program reading the file must endian convert all header 
        bytes to the endianness of the program (i.e. a little endian 
        program must convert from big endian, and a big endian program 
        must convert to little endian).
        """
        if endian == b'\x01\x02\x03\x04':
            self.little_endian = True
        elif endian == b'\x04\x03\x02\x01':
            self.little_endian = False
        else:
            raise KtxParseError('Unrecognized KTX endian specifier %s' % endian)
        # OpenGL texture metadata
        self.gl_type = self._read_uint32(file)
        # print (self.gl_type)
        self.gl_type_size = self._read_uint32(file)
        # print (self.gl_type_size)
        self.gl_format = self._read_uint32(file)
        # print (self.gl_format)
        self.gl_internal_format = self._read_uint32(file)
        # print (self.gl_internal_format)
        self.gl_base_internal_format = self._read_uint32(file)
        # print (self.gl_base_internal_format)
        self.pixel_width = self._read_uint32(file)
        # print (self.pixel_width)
        self.pixel_height = self._read_uint32(file)
        # print (self.pixel_height)
        self.pixel_depth = self._read_uint32(file)
        # print (self.pixel_depth)
        self.number_of_array_elements = self._read_uint32(file)
        # print (self.number_of_array_elements)
        self.number_of_faces = self._read_uint32(file)
        # print (self.number_of_faces)
        self.number_of_mipmap_levels = self._read_uint32(file)
        # print (self.number_of_mipmap_levels)
        # key,value metadata
        bytes_of_key_value_data = self._read_uint32(file) # bytesOfKeyValueData
        # print (bytes_of_key_value_data)
        remaining_key_value_bytes = bytes_of_key_value_data
        # print (bytes_of_key_value_data)
        self.key_value_metadata.clear()
        while remaining_key_value_bytes > 4:
            byte_size = self._read_uint32(file) # keyAndValueByteSize
            # print (byte_size)
            key_and_value = file.read(byte_size)
            padding = 3 - ((byte_size + 3) % 4)
            file.read(padding) # Value padding
            # print (padding)
            remaining_key_value_bytes -= byte_size
            remaining_key_value_bytes -= padding
            # Parse key and value
            key_end_idx = key_and_value.find(b'\x00')
            key = key_and_value[:key_end_idx]
            # print ("key = %s" % key)
            value = key_and_value[key_end_idx+1:]
            # print ("value = %s" % value)
            self.key_value_metadata[key] = value
        
    def write_stream(self, stream):
        # Identifier
        stream.write(b'\xabKTX 11\xbb\r\n\x1a\n')
        if self.little_endian:
            stream.write(b'\x01\x02\x03\x04')
        else:
            stream.write(b'\x04\x03\x02\x01')
        self._write_uint32(stream, self.gl_type)
        self._write_uint32(stream, self.gl_type_size)
        self._write_uint32(stream, self.gl_format)
        self._write_uint32(stream, self.gl_internal_format)
        self._write_uint32(stream, self.gl_base_internal_format)
        self._write_uint32(stream, self.pixel_width)
        self._write_uint32(stream, self.pixel_height)
        self._write_uint32(stream, self.pixel_depth)
        self._write_uint32(stream, self.number_of_array_elements)
        self._write_uint32(stream, self.number_of_faces)
        self._write_uint32(stream, self.number_of_mipmap_levels)
        # 
        key_values = io.BytesIO()
        for key, value in self.key_value_metadata.items():
            kv = io.BytesIO()
            kv.write(key)
            kv.write(b'\x00')
            try:
                kv.write(value)
            except TypeError:
                print (key, value)
                raise
            size = len(kv.getvalue())
            padding = 3 - ((size + 3) % 4)
            self._write_uint32(key_values, size) # keyAndValueByteSize
            key_values.write(kv.getvalue()) # keyAndValue
            key_values.write(padding * b'\x00') # valuePadding
        self._write_uint32(stream, len(key_values.getvalue())) # bytesOfKeyValueData
        stream.write(key_values.getvalue())
        
    def _read_uint32(self, stream):
        return struct.unpack(self._endian_char() + b'I', stream.read(4))[0]

    def _write_uint32(self, stream, val):
        stream.write(struct.pack(self._endian_char() + b'I', val))
        
    def _endian_char(self):
        if self.little_endian:
            return b'<'
        return b'>'

    
class KtxImageData(object):
    
    def __init__(self):
        self.mipmaps = list()
    
    def read_stream(self, file, header):
        for _ in range(_not_zero(header.number_of_mipmap_levels)):
            image_size = header._read_uint32(file)
            # print('Image size = %d' % image_size)
            if header.number_of_faces == 6 and header.number_of_array_elements == 0:
                raise # TODO - non-array cubemap case
            else:
                self.mipmaps.append(file.read(image_size))
            mip_padding_size = 3 - ((image_size + 3) % 4)
            file.read(mip_padding_size)
            
    def write_stream(self, file, header):
        for mipmap in self.mipmaps:
            if header.number_of_faces == 6 and header.number_of_array_elements == 0:
                raise # TODO - non-array cubemap case
            else: # typical non-cubemap case
                image_size = len(mipmap)
                header._write_uint32(file, image_size)
                file.write(mipmap)
                mip_padding_size = 3 - ((image_size + 3) % 4)
                file.write(mip_padding_size * b'\x00')
            

def _not_zero(val):
    if val == 0:
        return 1
    return val
