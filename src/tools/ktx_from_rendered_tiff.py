'''
Created on Sep 15, 2016

@author: brunsc
'''

import os
import posixpath
import glob


from small_memory_histogram import histogram_tiff_file


class RenderedMouseLightOctree(object):
    "Represents a folder containing an octree hierarchy of RenderedTiffBlock volume images"
    
    def __init__(self, folder):
        self.folder = folder

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
    "RenderedBlock represents one rendered Mouse Light volume image block"
    
    def __init__(self, folder):
        "Folder contains one or more 'default.0.tif', 'default.1.tif', etc. channel files"
        self.channel_files = glob.glob(posixpath.join(folder, "default.*.tif"))
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


def exercise_histogram():
    b = RenderedTiffBlock('.')
    b._populate_size_and_histograms()
    print (b.zyx_size, b.dtype)

def exercise_octree():
    o = RenderedMouseLightOctree(posixpath.abspath('.'))
    for b in o.iter_blocks(max_level=0):
        print (b.channel_files)
        b._populate_size_and_histograms()
        print (b.zyx_size, b.dtype)


if __name__ == '__main__':
    # exercise_histogram()
    exercise_octree()
