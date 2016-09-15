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
    for page in tif.iter_images():
        histogram_one_tiff_page(page, histogram)
        break # For debugging
    tif.close()
    return histogram

def histogram_one_tiff_page(page, histogram):
    h = numpy.histogram(page, bins=65536, range=(0, 65535), density=False)[0]
    histogram += h

if __name__ == '__main__':
    histogram = histogram_tiff_file('default.0.tif')
    print (histogram)
