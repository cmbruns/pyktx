'''
Created on Sep 15, 2016

@author: brunsc
'''

import os
import glob
import io
import re
import math
import datetime
from collections import deque

import numpy
import libtiff
from libtiff import TIFF

from small_memory_histogram import histogram_tiff_file
import ktx
from ktx import KtxHeader
from ktx.util import mipmap_shapes, _assort_subvoxels, _filter_assorted_array,\
    interleave_channel_arrays, downsample_array_xy, mipmap_dimension


class RenderedMouseLightOctree(object):
    "Represents a folder containing an octree hierarchy of RenderedTiffBlock volume images"
    
    def __init__(self, folder, mipmap_filter='arthur', downsample_xy=False, downsample_intensity=False):
        self.folder = folder
        # Parse the origin and voxel size from file "transform.txt" in the root folder
        self.transform = dict()
        with io.open(os.path.join(folder, "transform.txt"), 'r') as transform_file:
            for line in transform_file:
                fields = line.split(": ")
                if len(fields) != 2:
                    continue
                self.transform[fields[0].strip()] = fields[1].strip()
        # Convert nanometers to micrometers
        umFromNm = 1.0/1000.0
        self.origin_um = umFromNm * numpy.array([float(self.transform[key]) for key in ['ox', 'oy', 'oz']], dtype='float64')
        self.voxel_um = umFromNm * numpy.array([float(self.transform[key]) for key in ['sx', 'sy', 'sz']], dtype='float64')
        self.number_of_levels = int(self.transform['nl'])
        self.specimen_id = os.path.split(folder)[-1]
        self.mipmap_filter = mipmap_filter
        self.downsample_xy = downsample_xy
        self.downsample_intensity = downsample_intensity
        # Get base image size, so we can convert voxel size into total volume size
        temp_block = RenderedTiffBlock(folder, self, [])
        temp_block._populate_size()
        # Rearrange image size from zyx to xyz
        self.volume_voxels = numpy.array([temp_block.input_zyx_size[i] for i in [2, 1, 0]], dtype='uint32')
        # Compute total volume size in micrometers
        self.volume_um = self.voxel_um * self.volume_voxels
        #
        self.input_dtype = temp_block.input_dtype
        self.output_dtype = temp_block.input_dtype # Default to same data type for input and output
        if downsample_intensity and self.input_dtype.itemsize == 2:
            if self.input_dtype == numpy.uint16:
                self.output_dtype = numpy.dtype('uint8')
            else:
                raise NotImplementedError("unexpected data type " + str(self.input_dtype))
        self.input_zyx_size = temp_block.input_zyx_size
        self.output_zyx_size = self.input_zyx_size
        if self.downsample_xy:
            self.output_zyx_size = tuple(
                    [self.input_zyx_size[0],
                    mipmap_dimension(1, self.input_zyx_size[1]),
                    mipmap_dimension(1, self.input_zyx_size[2]),])

    def iter_blocks(self, max_level=None, folder=None):
        "Walk through rendered blocks, starting at folder folder, up to max_level steps deeper"
        if folder is None:
            folder = self.folder
        level = 0
        if level > max_level:
            return
        if not os.path.isdir(folder):
            return
        octree_path0 = os.path.relpath(folder, self.folder)
        octree_path0 = octree_path0.split('/')
        octree_path = []
        for level in octree_path0:
            if re.match(r'[1-8]', level):
                octree_path.append(int(level) - 1)
        yield RenderedTiffBlock(folder, self, octree_path)
        if level == max_level:
            return
        for subfolder in [str(i) for i in range(1, 9)]:
            # print (subfolder)
            for b in self.iter_blocks(max_level=max_level - 1, folder=os.path.join(folder, subfolder)):
                yield b


class RenderedTiffBlock(object):
    "RenderedBlock represents all channels of one rendered Mouse Light volume image block"
    
    def __init__(self, folder, octree_root, octree_path):
        "Folder contains one or more 'default.0.tif', 'default.1.tif', etc. channel files"
        self.folder = folder
        self.octree_root = octree_root
        self.octree_path = octree_path
        self.channel_files = glob.glob(os.path.join(folder, "default.*.tif"))
        self.input_zyx_size = None

    def _populate_size(self):
        """
        Cheapest pre-pass to get just the image size
        """
        channel = RTBChannel(self.channel_files[0])
        channel._populate_size_and_histogram()
        self.input_zyx_size = channel.input_zyx_size
        self.input_dtype = channel.input_dtype
        self.output_type = self.input_dtype

    def _populate_size_and_histograms(self):
        """
        First pass of small-memory tile processing.
        Read through channel tiff files, storing image size and per-channel 
        intensity histogram.
        """
        self.channels = []
        for fname in self.channel_files:
            channel = RTBChannel(fname)
            channel._populate_size_and_histogram()
            if self.input_zyx_size is None:
                self.input_zyx_size = channel.input_zyx_size
                self.input_dtype = channel.input_dtype
            else:
                assert self.input_zyx_size == channel.input_zyx_size # All channel files must be the same size
                assert self.input_dtype == channel.input_dtype
            self.channels.append(channel)
        # Prepare to create first part of ktx file
        self.ktx_header = KtxHeader()
        self.mipmap_shapes = mipmap_shapes(self.octree_root.output_zyx_size)
        self.ktx_header.populate_from_array_params(
                shape=self.octree_root.output_zyx_size, 
                dtype=self.octree_root.output_dtype, 
                channel_count=len(self.channels))
        self._populate_octree_metadata()
        self._mipmap_parent_slice_cache = [deque() for _ in range(len(self.mipmap_shapes))] # Holds up to three recent slices at each mipmap level
        assert len(self._mipmap_parent_slice_cache) == len(self.mipmap_shapes)
        self._mipmap_slice_cache = [list() for _ in range(len(self.mipmap_shapes))] # Holds all mipmap slices from levels one and higher
        
    def _populate_octree_metadata(self):
        kh = self.ktx_header
        kh["distance_units"] = "micrometers"
        kh['multiscale_level_id'] = len(self.octree_path)
        kh['multiscale_total_levels'] = self.octree_root.number_of_levels
        kh['octree_path'] = "/".join(self.octree_path)
        assert kh['multiscale_level_id'] < kh['multiscale_total_levels']
        # Walk octree path to compute geometric parameters. Because someone
        # decided that having a separate transform.txt in each folder was
        # "inefficient" and implemented it that way in production one day as a 
        # fait accompli.
        # Initialize with top-level origin and size
        self.origin_um = numpy.array(self.octree_root.origin_um, copy=True)
        self.volume_um = numpy.array(self.octree_root.volume_um, copy=True)
        for level in self.octree_path: # range 0-7
            self.volume_um *= 0.5 # Every deeper level is half the size of the lower level
            # Shift origin if this is a right/bottom/far sub-block
            bigZ = level > 4 # 4,5,6,7
            bigX = level % 2 > 0 # 1,3,5,7
            bigY = level in [2,3,6,7] # 2,3,6,7
            if bigZ: # if Z is large, shift Z origin
                self.origin_um[2] += self.volume_um[2]
            if bigY:
                self.origin_um[1] += self.volume_um[1]
            if bigX:
                self.origin_um[0] += self.volume_um[0]
        # Compute linear transform from texture coordinates to microscope stage coordinates
        ox, oy, oz = self.origin_um
        sx, sy, sz = self.volume_um
        xform = numpy.array([
                [sx, 0, 0, ox],
                [0, sy, 0, oy],
                [0, 0, sz, oz],
                [0, 0, 0, 1],], dtype='float64')
        kh["xyz_from_texcoord_xform"] = xform
        center = numpy.array( (ox + 0.5*sx, oy + 0.5*sy, oz + 0.5*sz,), )
        radius = math.sqrt(sx*sx + sy*sy + sz*sz)/16.0
        kh['bounding_sphere_center'] = center
        kh['bounding_sphere_radius'] = radius        
        # Nominal resolution
        resX = sx / kh.pixel_width
        resY = sy / kh.pixel_height
        resZ = sz / kh.pixel_depth
        rms = math.sqrt(numpy.mean(numpy.square([resX, resY, resZ],)))
        kh['nominal_resolution'] = rms
        # Specimen ID
        kh['specimen_id'] = self.octree_root.specimen_id
        # Relation to parent tile/block
        kh['mipmap_filter'] = self.octree_root.mipmap_filter
        relations = list()
        if self.octree_root.downsample_xy:
            relations.append("downsampled 2X in X & Y")
        if self.octree_root.downsample_intensity:
            relations.append("rescaled intensity to 8 bits")
        if len(relations) == 0:
            relations.append("unchanged")
        kh['relation_to_parent'] = ";".join(relations)
        # TODO: Per channel statistics
        kh['ktx_file_creation_date'] = datetime.datetime.now()
        # print (kh['ktx_file_creation_date'])
        import __main__ #@UnresolvedImport
        kh['ktx_file_creation_program'] = __main__.__file__
        # print (kh['ktx_file_creation_program'])
        kh['pyktx_version'] = ktx.__version__

    def _process_mipmap_parent_slice(self, mipmap_level, parent_z_index, channel_slices):
        "Recursive method to incrementally update deeper mipmaps, one slice at a time"
        if mipmap_level >= len(self.mipmap_shapes):
            # Parent is presumably the deepest mipmap level
            assert parent_z_index == 0
            assert channel_slices[0].shape == (1,1)
            return # Too Deep. No mipmap here
        # print("Processing mipmap level %d, slice %d" % (mipmap_level-1, parent_z_index))
        # Maintain a deque of up to three parent slices, to save memory
        parent_slices = self._mipmap_parent_slice_cache[mipmap_level]
        parent_slices.append(channel_slices)
        if len(parent_slices) > 3:
            parent_slices.popleft() # For small-memory, only keep a maximum of three parent slices queued
        # Determine whether we are ready to create a new mipmap slice
        daughter_slices = self._mipmap_slice_cache[mipmap_level]
        next_daughter_slice_index = len(daughter_slices)
        needed_parent_slice_index = next_daughter_slice_index * 2 + 1 # zero-based, assuming two parent slices per daughter slice
        parent_sz = self.mipmap_shapes[mipmap_level - 1][0]
        needed_parent_slice_index = min(parent_sz-1, needed_parent_slice_index) # For single-slice parents
        daughter_sz = self.mipmap_shapes[mipmap_level][0]
        use_thick_middle_slice = False
        if parent_sz % 2 > 0 and parent_sz >= 3: # Odd parent z-dimension
            if next_daughter_slice_index >= daughter_sz // 2:
                needed_parent_slice_index += 1 # Second half of stack reads one slice later
                if next_daughter_slice_index >= daughter_sz // 2:
                    use_thick_middle_slice = True # This exact slice is the fat one in the downsampling stack
        if parent_z_index < needed_parent_slice_index:
            return
        assert parent_z_index == needed_parent_slice_index
        daughter_channel_slices = []
        # Downsample parent_slices to daughter_slices
        for channel in range(len(channel_slices)):
            if use_thick_middle_slice: # Special thick slice in the very middle uses three parent slices
                parent_channels = [parent_slices[i][channel] for i in [-3,-2,-1,]]
            elif parent_sz < 2:
                parent_channels = [parent_slices[i][channel] for i in [-1,]] # One slice: sometimes there is only one
            else:
                parent_channels = [parent_slices[i][channel] for i in [-2,-1,]] # Usually we combine two parent slices
            desired_shape = list(self.mipmap_shapes[mipmap_level])
            desired_shape[0] = 1 # Just one slice, please, but still a 3D shape, for the moment
            desired_shape = tuple(desired_shape)
            arr = numpy.array(parent_channels, dtype=self.octree_root.output_dtype, copy=True)
            scratch = _assort_subvoxels(arr, desired_shape)
            new_slice = _filter_assorted_array(scratch, self.octree_root.mipmap_filter)
            assert new_slice.shape == desired_shape
            # Reshape again, to remove singleton z-dimension, before storing in slice array
            shape_2d = tuple(desired_shape[1:])
            new_slice.shape = shape_2d
            assert new_slice.shape == shape_2d
            daughter_channel_slices.append(new_slice)
        daughter_slices.append(daughter_channel_slices)
        # Recurse to deeper mipmap level
        self._process_mipmap_parent_slice(mipmap_level + 1, next_daughter_slice_index, daughter_channel_slices)

    def _process_tiff_slice(self, z_index, channel_slices, output_stream):
        """
        Interleave individual color channels into one multicolor slice.
        And also accumulate deeper mipmaps levels.
        """
        zslice0 = interleave_channel_arrays(channel_slices)
        # Save this slice to ktx file on disk
        data0 = zslice0.tostring()
        image_size = len(data0) * self.mipmap_shapes[0][0]
        if z_index == 0: # Write total number of bytes for this mipmap before first slice
            self.ktx_header._write_uint32(output_stream, image_size)
        output_stream.write(data0)
        # Propagate deeper mipmap construction
        self._process_mipmap_parent_slice(mipmap_level=1, parent_z_index=z_index, channel_slices=channel_slices)
        return image_size
    
    def _stream_first_mipmap(self, stream, filter_='arthur'):
        """
        Small-memory implementation for streaming first mipmap from TIFF channel files to KTX file.
        Simultaneously accumulates deeper mipmap levels in memory, for later streaming.
        """
        # 1) Open all the channel tiff files
        # channel_iterators = []
        tif_streams = []
        for channel in self.channels:
            tif = TIFF.open(channel.file_name, mode='r')
            channel.tif_iterator = tif.iter_images()
            tif_streams.append(tif)
        # 2) Load and process one z-slice at a time
        zslice_shape0 = self.mipmap_shapes[0][1:3] # for sanity checking
        sz = self.input_zyx_size[0]
        for z_index in range(sz):
            channel_slices = [] # For level zero mipmap
            for channel in self.channels:
                zslice = next(channel.tif_iterator)
                # Process slice, if necessary
                # 1) Spatial downsampling in XY
                if self.octree_root.downsample_xy:
                    zslice.shape = tuple([1, zslice.shape[0], zslice.shape[1],]) # reshape to include unity z dimension
                    zslice = downsample_array_xy(zslice, self.octree_root.mipmap_filter)
                    zslice.shape = zslice.shape[1:] # Back to 2D
                # 2) Intensity downsamping
                if self.octree_root.downsample_intensity and self.input_dtype != self.octree_root.output_dtype:
                    black_level = channel.downsample_intensity_params[0]
                    white_level = channel.downsample_intensity_params[1]
                    gamma = channel.downsample_intensity_params[2]
                    zslice1 = numpy.array(zslice, dtype='float64', copy=True)
                    zslice1 -= black_level # Set lower bound to zero 
                    zslice1[zslice1 <= 1] = 1 # Truncate small values to 1
                    zslice1[zslice == 0] = 0 # Reset all "no data" voxels back to zero
                    range_ = float(white_level - black_level)
                    zslice1 *= 1.0 / range_ # Scale to range 0 - 1
                    zslice1[zslice1 >= 1.0] = 1.0 # Truncate large values to 1.0
                    zslice1 = zslice1 ** gamma # Gamma correct to emphasize dim intensities
                    zslice1 *= 255.0 # Restore to range 0-255
                    zslice1 = numpy.ceil(zslice1) # Ensure small finite values are at least 1.0
                    zslice = numpy.array(zslice1, dtype=self.octree_root.output_dtype)
                    # TODO: populate KTX header with unmixing parameters
                assert zslice.shape == zslice_shape0
                channel_slices.append(zslice)
            image_size_bytes = self._process_tiff_slice(z_index, channel_slices, stream)
        # Pad final bytes of mipmap in output file
        mip_padding_size = 3 - ((image_size_bytes + 3) % 4)
        stream.write(mip_padding_size * b'\x00')
        # 3) Close all the input tif files
        for tif in tif_streams:
            tif.close()
            
    def _stream_other_mipmaps(self, stream):
        for mipmap_level in range(1, len(self.mipmap_shapes)):
            daughter_slices = self._mipmap_slice_cache[mipmap_level]
            for z_index in range(len(daughter_slices)):
                zslice_channels = daughter_slices[z_index]
                zslice = interleave_channel_arrays(zslice_channels)
                # Save this slice to ktx file on disk
                slice_bytes = zslice.tostring()
                image_size_bytes = len(slice_bytes) * self.mipmap_shapes[mipmap_level][0]
                if z_index == 0: # Write total number of bytes for this mipmap before first slice
                    self.ktx_header._write_uint32(stream, image_size_bytes)
                stream.write(slice_bytes)
            # Pad final bytes of mipmap in output file
            mip_padding_size = 3 - ((image_size_bytes + 3) % 4)
            stream.write(mip_padding_size * b'\x00')
    
    def write_ktx_file(self, stream):
        if self.input_zyx_size is None:
            self._populate_size_and_histograms()
        self.ktx_header.number_of_mipmap_levels = len(self.mipmap_shapes)
        self.ktx_header.write_stream(stream)
        # Stream from input TIFF files, while writing first mipmap to ktx
        self._stream_first_mipmap(stream)
        self._stream_other_mipmaps(stream)


class RTBChannel(object):
    """
    RTBChannel represents one channel of a RenderedTiffBlock
    """
    def __init__(self, file_name):
        self.file_name = file_name
        
    def _populate_size(self):
        """
        Minimal parsing of size and type
        """
        tif = TIFF.open(self.file_name, mode='r')
        sz = 0
        sxy = None
        dtype = None
        for page in tif.iter_images():
            sy = page.shape[0]
            sx = page.shape[1]
            if sxy is None:
                sxy = tuple([sx, sy,])
                dtype = page.dtype
            else:
                assert sxy == tuple([sx, sy,]) # All slices must be the same size
                assert page.dtype == dtype
            sz += 1
        tif.close()
        size = tuple([sz, sy, sx,])
        self.input_zyx_size = size
        self.input_dtype = dtype

    def _populate_size_and_histogram(self):
        """
        First pass of small-memory tile processing.
        Read through one channel tiff file, storing image size and
        intensity histogram.
        """
        self.input_zyx_size, self.histogram, self.input_dtype = histogram_tiff_file(self.file_name)
        # Create histogram of non-zero intensities (because zero means "no data"
        self.percentiles = numpy.zeros((101,), dtype='uint32')
        total_non_zero = 0
        min_non_zero = 0
        max_non_zero = 0
        for i in range(1, 65536):
            count = self.histogram[i]
            if count == 0:
                continue
            total_non_zero += count
            if min_non_zero == 0:
                min_non_zero = i
            max_non_zero = i
        # print("Total non-zero intensity voxel count = ", total_non_zero)
        # print("Total zero intensity voxel count = ", self.histogram[0])
        accumulated = 0
        percentage = 0.0
        # print(0, min_non_zero)
        for i in range(1, 65536):
            floor_percentage = percentage
            accumulated += self.histogram[i]
            ceil_percentage = 100.0 * accumulated / float(total_non_zero);
            percentage = ceil_percentage
            min_bin = int(floor_percentage)
            max_bin = int(ceil_percentage)
            if min_bin == max_bin:
                continue
            for p in range(min_bin+1, max_bin+1):
                self.percentiles[p] = i
                # print(p, i)
        # print(100, max_non_zero)
        self.percentiles[0] = min_non_zero
        self.percentiles[100] = max_non_zero
        # Print histogram of incremental percentiles
        for i in range(1, 101):
            pass
            # print(i, self.percentiles[i] - self.percentiles[i-1], self.percentiles[i])
        self.downsample_intensity_params = self._compute_intensity_downsample_params()
        # print(self.downsample_intensity_params)
            
    def _compute_intensity_downsample_params(self, min_quantile=20, max_base_quantile=90, max_sigma_buffer=6.0):
        """
        Use internal histogram data to estimate optimal sparse neuron intensity downsampling.
        Input parameters:
            min_quantile: the dimmest <min_quantile> percent of intensities will be truncated to value "1"
            max_base_quantile: intensities above the intensity at <max_base_quantile> plus max_sigma_buffer 
                standard deviations, will be truncated to value 255.
        Output:
            Three downsampling parameters:
                black_level
                white_level
                gamma
                result(I8) = (((I16 - offset) * 1/(black_level - white_level)) ^ gamma) * 255
        """
        # Compute intensity statistics in relevant range
        mean_intensity = 0
        for p in range(min_quantile, max_base_quantile+1):
            mean_intensity += self.percentiles[p]
        mean_intensity /= float(max_base_quantile - min_quantile + 1)
        variance = 0
        for p in range(min_quantile, max_base_quantile+1):
            d = self.percentiles[p] - mean_intensity
            variance += d*d
        variance /= float(max_base_quantile - min_quantile + 1)
        stddev = math.sqrt(variance)
        print("Mean = %.1f, stddev = %.1f" % (mean_intensity, stddev))
        black_level = self.percentiles[min_quantile]
        white_level = int(self.percentiles[max_base_quantile] + max_sigma_buffer * stddev)
        white_level = min(white_level, self.percentiles[100]) # Don't go above max intensity
        gamma = 0.5
        return black_level, white_level, gamma
        

def _exercise_histogram():
    "for testing histogram construction during developement"
    b = RenderedTiffBlock('.')
    b._populate_size_and_histograms()
    # print (b.input_zyx_size, b.dtype)

def _exercise_octree():
    "for testing octree walking during development"
    o = RenderedMouseLightOctree(os.path.abspath('./practice_octree_input'), 
            downsample_intensity=False,
            downsample_xy=True)
    # Visit top layer of the octree
    for b in o.iter_blocks(max_level=0):
        print (b.channel_files)
        b._populate_size_and_histograms()
        print (b.input_zyx_size, b.input_dtype)
        f = open('./practice_octree_output/test.ktx', 'wb')
        b.write_ktx_file(f)


if __name__ == '__main__':
    libtiff.libtiff_ctypes.suppress_warnings()
    # exercise_histogram()
    _exercise_octree()
