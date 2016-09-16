'''
Created on Sep 15, 2016

@author: brunsc
'''

import os
import glob
import io

import numpy

from small_memory_histogram import histogram_tiff_file
from ktx import KtxHeader


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
        self.ktx_header = KtxHeader()
        self.ktx_header.populate_from_array_params(
                shape=self.zyx_size, 
                dtype=self.dtype, 
                channel_count=len(self.channels))


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
            print(i-1, i, self.percentiles[i] - self.percentiles[i-1], self.percentiles[i])

def exercise_histogram():
    b = RenderedTiffBlock('.')
    b._populate_size_and_histograms()
    print (b.zyx_size, b.dtype)

def exercise_octree():
    o = RenderedMouseLightOctree(os.path.abspath('./practice_octree_input'))
    # Visit top layer of the octree
    for b in o.iter_blocks(max_level=0):
        print (b.channel_files)
        b._populate_size_and_histograms()
        print (b.zyx_size, b.dtype)


if __name__ == '__main__':
    # exercise_histogram()
    exercise_octree()
