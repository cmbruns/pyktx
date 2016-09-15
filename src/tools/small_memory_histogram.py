'''
small_memory_histogram.py

proof-of-concept for first pass of small-memory volume processing

Created on Sep 15, 2016

@author: brunsc
'''

import numpy
# NOTE: If necessary, get latest python 3 compatible libtiff module from
#  https://github.com/pearu/pylibtiff
from libtiff import TIFF


def histogram_tiff_file(file_name):
    histogram = numpy.zeros(shape=(65536), dtype='int32')
    tif = TIFF.open(file_name, mode='r')
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
        histogram_one_tiff_page(page, histogram)
        sz += 1
    tif.close()
    size = tuple([sz, sy, sx,])
    return size, histogram, dtype

def histogram_one_tiff_page(page, histogram):
    h = numpy.histogram(page, bins=65536, range=(0, 65535), density=False)[0]
    histogram += h


if __name__ == '__main__':
    histogram = histogram_tiff_file('default.0.tif')[1]
    print (histogram)
