"""
https://www.khronos.org/opengles/sdk/tools/KTX/file_format_spec/
"""

import io
import struct


class Ktx(object):
    
    def read_filename(self, file_name):
        with io.open(file_name, 'rb') as fh:
            self.read_stream(fh)
    
    def read_stream(self, file):
        self.header = KtxHeader()
        self.header.read_stream(file)
        self.image_data = KtxImageData()
        self.image_data.read_stream(file, self.header)
        
    def write_stream(self, file):
        self.header.write_stream(file)
        self.image_data.write_stream(file, self.header)
        

class KtxParseError(Exception):
    pass


class KtxHeader(object):
    
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
            self.endian_char = b'<'
        elif endian == b'\x04\x03\x02\x01':
            self.little_endian = False
            self.endian_char = b'>'
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
        self.key_value_metadata = dict()
        self.keys = list()
        remaining_key_value_bytes = bytes_of_key_value_data
        # print (bytes_of_key_value_data)
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
            self.keys.append(key)
        
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
        for key in self.keys:
            kv = io.BytesIO()
            kv.write(key)
            kv.write(b'\x00')
            kv.write(self.key_value_metadata[key])
            size = len(kv.getvalue())
            padding = 3 - ((size + 3) % 4)
            self._write_uint32(key_values, size) # keyAndValueByteSize
            key_values.write(kv.getvalue()) # keyAndValue
            key_values.write(padding * b'\x00') # valuePadding
        self._write_uint32(stream, len(key_values.getvalue())) # bytesOfKeyValueData
        stream.write(key_values.getvalue())
        
    def _read_uint32(self, stream):
        return struct.unpack(self.endian_char + b'I', stream.read(4))[0]

    def _write_uint32(self, stream, val):
        stream.write(struct.pack(self.endian_char + b'I', val))

    
class KtxImageData(object):
    
    def __init__(self):
        self.mipmaps = list()
    
    def read_stream(self, file, header):
        for mipmap_level in range(_not_zero(header.number_of_mipmap_levels)):
            image_size = header._read_uint32(file)
            # print('Image size = %d' % image_size)
            if header.number_of_faces == 6 and header.number_of_array_elements == 0:
                pass # TODO - non-array cubemap case
            else:
                self.mipmaps.append(file.read(image_size))
            mip_padding_size = 3 - ((image_size + 3) % 4)
            file.read(mip_padding_size)
            
    def write_stream(self, file, header):
        for mipmap in self.mipmaps:
            if header.number_of_faces == 6 and header.number_of_array_elements == 0:
                pass # TODO - non-array cubemap case
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
