'''
Created on Sep 15, 2016

@author: brunsc
'''

import os
import glob
import io

import numpy
from libtiff import TIFF

from small_memory_histogram import histogram_tiff_file
from ktx import KtxHeader
from ktx.util import mipmap_shapes, _assort_subvoxels, _filter_assorted_array,\
    interleave_channel_arrays


class RenderedMouseLightOctree(object):
    "Represents a folder containing an octree hierarchy of RenderedTiffBlock volume images"
    
    def __init__(self, folder):
        self.folder = folder
        self.transform = dict()
        with io.open(os.path.join(folder, "transform.txt"), 'r') as transform_file:
            for line in transform_file:
                fields = line.split(": ")
                if len(fields) != 2:
                    continue
                self.transform[fields[0].strip()] = fields[1].strip()

    def iter_blocks(self, max_level=None, folder=None):
        "Walk through rendered blocks, starting at folder folder, up to max_level steps deeper"
        if folder is None:
            folder = self.folder
        level = 0
        if level > max_level:
            return
        if not os.path.isdir(folder):
            return
        yield RenderedTiffBlock(folder)
        if level == max_level:
            return
        for subfolder in [str(i) for i in range(1, 9)]:
            print (subfolder)
            for b in self.iter_blocks(max_level=max_level - 1, folder=os.path.join(folder, subfolder)):
                yield b


class RenderedTiffBlock(object):
    "RenderedBlock represents all channels of one rendered Mouse Light volume image block"
    
    def __init__(self, folder):
        "Folder contains one or more 'default.0.tif', 'default.1.tif', etc. channel files"
        self.channel_files = glob.glob(os.path.join(folder, "default.*.tif"))
        self.zyx_size = None

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
