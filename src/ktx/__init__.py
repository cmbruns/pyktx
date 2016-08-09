"""
https://www.khronos.org/opengles/sdk/tools/KTX/file_format_spec/
"""

import io
import struct


class Ktx(object):
    
    def load_filename(self, file_name):
        with io.open(file_name, 'rb') as fh:
            self.load_stream(fh)
    
    def load_stream(self, file):
        self.header = KtxHeader()
        self.header.load_stream(file)
        self.image_data = KtxImageData()
        self.image_data.load_stream(file, self.header)
        

class KtxParseError(Exception):
    pass


class KtxHeader(object):
    
    def load_stream(self, file):
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
        print (self.gl_type)
        self.gl_type_size = self._read_uint32(file)
        print (self.gl_type_size)
        self.gl_format = self._read_uint32(file)
        print (self.gl_format)
        self.gl_internal_format = self._read_uint32(file)
        print (self.gl_internal_format)
        self.gl_base_internal_format = self._read_uint32(file)
        print (self.gl_base_internal_format)
        self.pixel_width = self._read_uint32(file)
        print (self.pixel_width)
        self.pixel_height = self._read_uint32(file)
        print (self.pixel_height)
        self.pixel_depth = self._read_uint32(file)
        print (self.pixel_depth)
        self.number_of_array_elements = self._read_uint32(file)
        print (self.number_of_array_elements)
        self.number_of_faces = self._read_uint32(file)
        print (self.number_of_faces)
        self.number_of_mipmap_levels = self._read_uint32(file)
        print (self.number_of_mipmap_levels)
        # key,value metadata
        self.bytes_of_key_value_data = self._read_uint32(file)
        print (self.bytes_of_key_value_data)
        self.key_value_metadata = dict()
        remaining_key_value_bytes = self.bytes_of_key_value_data
        while remaining_key_value_bytes > 0:
            byte_size = self._read_uint32(file)
            key_and_value = file.read(byte_size)
            value_padding = file.read(3 - ((byte_size + 3) % 4))
            remaining_key_value_bytes -= byte_size
            # Parse key and value
            key_end_idx = key_and_value.find('\x00')
            key = key_and_value[:key_end_idx]
            value = key_and_value[key_end_idx+1:]
            self.key_value_metadata[key] = value
        
    def _read_uint32(self, stream):
        return struct.unpack(self.endian_char + b'I', stream.read(4))[0]

    
class KtxImageData(object):
    
    def load_stream(self, file, header):
        for mipmap_level in range(_not_zero(header.number_of_mipmap_levels)):
            for array_element in range(_not_zero(header.number_of_array_elements)):
                for face in range(header.number_of_faces):
                    for z_slice in range(_not_zero(header.pixel_depth)):
                        for row in range(_not_zero(header.pixel_height)):
                            for pixel in range(header.pixel_width):
                                pass # TODO read bytes
                    # TODO cube padding
            # TODO mipmap padding                    


def _not_zero(val):
    if val == 0:
        return 1
    return val