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

import numpy
from libtiff import TIFF

from small_memory_histogram import histogram_tiff_file
import ktx
from ktx import KtxHeader
from ktx.util import mipmap_shapes, _assort_subvoxels, _filter_assorted_array,\
    interleave_channel_arrays


class RenderedMouseLightOctree(object):
    "Represents a folder containing an octree hierarchy of RenderedTiffBlock volume images"
    
    def __init__(self, folder):
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
        # Get base image size, so we can convert voxel size into total volume size
        temp_block = RenderedTiffBlock(folder, self, [])
        temp_block._populate_size_and_histograms()
        # Rearrange image size from zyx to xyz
        self.volume_voxels = numpy.array([temp_block.zyx_size[i] for i in [2, 1, 0]], dtype='uint32')
        # Compute total volume size in micrometers
        self.volume_um = self.voxel_um * self.volume_voxels

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
            print (subfolder)
            for b in self.iter_blocks(max_level=max_level - 1, folder=os.path.join(folder, subfolder)):
                yield b


class RenderedTiffBlock(object):
    "RenderedBlock represents all channels of one rendered Mouse Light volume image block"
    
    def __init__(self, folder, octree_root, octree_path, mipmap_filter='arthur', downsample_xy=False, downsample_intensity=False):
        "Folder contains one or more 'default.0.tif', 'default.1.tif', etc. channel files"
        self.folder = folder
        self.octree_root = octree_root
        self.octree_path = octree_path
        self.channel_files = glob.glob(os.path.join(folder, "default.*.tif"))
        self.zyx_size = None
        self.downsample_xy = downsample_xy
        self.downsample_intensity = downsample_intensity
        self.mipmap_filter = mipmap_filter

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
            if self.zyx_size is None:
                self.zyx_size = channel.zyx_size
                self.dtype = channel.dtype
            else:
                assert self.zyx_size == channel.zyx_size # All channel files must be the same size
                assert self.dtype == channel.dtype
            self.channels.append(channel)
        # Prepare to create first part of ktx file
        self.ktx_header = KtxHeader()
        self.ktx_header.populate_from_array_params(
                shape=self.zyx_size, 
                dtype=self.dtype, 
                channel_count=len(self.channels))
        self._populate_octree_metadata()
        
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
        self.volume_um = numpy.array(self.octree_root.origin_um, copy=True)
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
        kh['mipmap_filter'] = self.mipmap_filter
        relations = list()
        if self.downsample_xy:
            relations.append("downsampled 2X in X & Y")
        if self.downsample_intensity:
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

    def _process_tiff_slice(self, z_index, channel_slices, output_stream):
        # Interleave individual color channels into one multicolor slice
        zslice0 = interleave_channel_arrays(channel_slices)
        # Save this slice to ktx file on disk
        data0 = zslice0.tostring()
        image_size = len(data0) * self.mipmap_shapes[0][0]
        if z_index == 0: # Write total number of bytes for this mipmap before first slice
            self.ktx_header._write_uint32(output_stream, image_size)
        output_stream.write(data0)
        return image_size
    
    def _stream_first_mipmap(self, stream, filter_='arthur'):
        "small-memory implementation for streaming first mipmap from TIFF channel files to KTX file"
        # 1) Open all the channel tiff files
        channel_iterators = []
        tif_streams = []
        for channel in self.channels:
            tif = TIFF.open(channel.file_name, mode='r')
            channel_iterators.append(tif.iter_images())
            tif_streams.append(tif)
        # 2) Load and process one z-slice at a time
        zslice_shape0 = self.mipmap_shapes[0][1:3] # for sanity checking
        zslice_shape1 = self.mipmap_shapes[1][1:3]
        # Working version of next level mipmap with have too many slices at first
        mipmap1_shape = list(self.mipmap_shapes[1])
        sz = self.zyx_size[0]
        mipmap1_shape[0] = sz
        mipmap1_shape.append(len(self.channels))
        # TODO: This bulk allocation might be the next memory bottleneck
        mipmap1 = numpy.zeros(shape=mipmap1_shape, dtype=self.dtype) # save second mipmap level while we are there
        for z_index in range(sz):
            channel_slices = [] # For level zero mipmap
            smaller_channel_slices = [] # For level one mipmap
            for channel in channel_iterators:
                zslice = next(channel)
                # TODO: process slice, if necessary
                assert zslice.shape == zslice_shape0
                channel_slices.append(zslice)
                # Downsample XY for next deeper mipmap level
                scratch = _assort_subvoxels(zslice, zslice_shape1)
                smaller_zslice = _filter_assorted_array(scratch, filter_)
                assert smaller_zslice.shape == zslice_shape1
                smaller_channel_slices.append(smaller_zslice)
            image_size = self._process_tiff_slice(z_index, channel_slices, stream)
            # TODO: might need to keep second mipmap channels separate until it's compacted to final size
            zslice1 = interleave_channel_arrays(smaller_channel_slices)
            mipmap1[z_index,:,:,:] = zslice1
            # TODO - store second mipmap slices somewhere
        # Pad final bytes of mipmap in output file
        mip_padding_size = 3 - ((image_size + 3) % 4)
        stream.write(mip_padding_size * b'\x00')
        # 3) Close all the input tif files
        for tif in tif_streams:
            tif.close()
        # 4) Consolidate second mipmap
        # TODO:
    
    def write_ktx_file(self, stream):
        if self.zyx_size is None:
            self._populate_size_and_histograms()
        output_shape = self.zyx_size # TODO: consider options
        self.mipmap_shapes = mipmap_shapes(output_shape)
        self.ktx_header.number_of_mipmap_levels = len(self.mipmap_shapes)
        self.ktx_header.write_stream(stream)
        self._stream_first_mipmap(stream)
        # TODO: stream from input TIFF files, while writing first mipmap to ktx


class RTBChannel(object):
    """
    RTBChannel represents one channel of a RenderedTiffBlock
    """
    def __init__(self, file_name):
        self.file_name = file_name
        
    def _populate_size_and_histogram(self):
        """
        First pass of small-memory tile processing.
        Read through one channel tiff file, storing image size and
        intensity histogram.
        """
        self.zyx_size, self.histogram, self.dtype = histogram_tiff_file(self.file_name)
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
        print("Total non-zero intensity voxel count = ", total_non_zero)
        print("Total zero intensity voxel count = ", self.histogram[0])
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
            print(i, self.percentiles[i] - self.percentiles[i-1], self.percentiles[i])

def _exercise_histogram():
    "for testing histogram construction during developement"
    b = RenderedTiffBlock('.')
    b._populate_size_and_histograms()
    print (b.zyx_size, b.dtype)

def _exercise_octree():
    "for testing octree walking during development"
    o = RenderedMouseLightOctree(os.path.abspath('./practice_octree_input'))
    # Visit top layer of the octree
    for b in o.iter_blocks(max_level=0):
        print (b.channel_files)
        b._populate_size_and_histograms()
        print (b.zyx_size, b.dtype)
        f = open('./practice_octree_output/test.ktx', 'wb')
        b.write_ktx_file(f)


if __name__ == '__main__':
    # exercise_histogram()
    _exercise_octree()
