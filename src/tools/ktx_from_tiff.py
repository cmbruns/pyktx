#!/bin/env python

import sys
from glob import glob
from tifffile import TiffFile
import tifffile
import numpy
from ktx import Ktx
from OpenGL import GL
import io
import lz4
import math

"""
TODO: For converting rendered octree blocks, include the following precomputed:
  * all mipmap levels
  * optional intensity downsampling, with affine reestimation parameters
  * optional spatial downsampling
  * other metadata:
      * distance units e.g. "micrometers", for all the transforms below
      * transform from texture coordinates to Cartesian reference space
      * Optional transform from texture coordinates to Allen reference space
      * center xyz in reference space
      * bounding radius
      * nominal spatial resolution range at this level
      * specimen ID, e.g. "2015-06-19-johan-full"
      * parent tile/block ID, e.g. "/1/5/4/8/default.[0,1].tif"
      * relation to parent tile/block, e.g. "downsampled 2X in XY; rescaled intensity to 8 bits; sub-block (1,2) of (6,6)
      * multiscale level ID (int)
          * of total multiscale level count (int)
      * per channel
          * affine parameters to approximate background level of first channel, for dynamic unmixing
          * min, max, average, median intensities
          * proportion of zero/NaN in this block
      * creation time
      * name of program used to create this block
      * version of program used to create this block
      * texture coordinate bounds for display (there might be padding...)
"""

def test_mipmap_dimension():
    assert mipmap_dimension(level=0, full=0) == 1
    assert mipmap_dimension(level=100, full=0) == 1
    assert mipmap_dimension(level=100, full=100) == 1
    assert mipmap_dimension(level=0, full=256) == 256
    assert mipmap_dimension(level=1, full=256) == 128
    assert mipmap_dimension(level=2, full=256) == 64
    assert mipmap_dimension(level=3, full=256) == 32
    assert mipmap_dimension(level=4, full=256) == 16
    assert mipmap_dimension(level=5, full=256) == 8
    assert mipmap_dimension(level=6, full=256) == 4
    assert mipmap_dimension(level=7, full=256) == 2
    assert mipmap_dimension(level=8, full=256) == 1
    assert mipmap_dimension(level=9, full=256) == 1
    assert mipmap_dimension(level=20, full=256) == 1
    assert mipmap_dimension(level=0, full=3) == 3
    assert mipmap_dimension(level=1, full=3) == 1
    assert mipmap_dimension(level=2, full=3) == 1
    

def mipmap_dimension(level, full):
    # Computes integer edge dimension for a mipmap of a particular level, based on the full sized (level zero) dimension
    return int(max(1, math.floor(full / 2**level)))

def test_downsample_xy():
    a1 = numpy.array( ( ((1,2,),(3,4,)),
                        ((5,6,),(7,8,)) ), 
                     dtype='uint16' )
    print (a1)
    print (a1.shape)
    print (numpy.percentile(a1, 99, interpolation='lower')) # Arthur filtering
    # TODO - reshape so clusters of 4 are in their own dimension
    # TODO - test with NaNs/zeros

def downsample_xy(arr):
    "downsample in X and Y directions, using second largest non-zero intensity"
    # TODO

def interleave_channel_arrays(arrays):
    "Combine multiple single channel stacks into one multi-channel stack"
    a = arrays[0]
    shp = list(a.shape)
    print (shp)
    print (len(shp))
    shp.append(len(arrays))
    print(shp)
    print (a.dtype)
    c = numpy.empty( shape=shp, dtype=a.dtype)
    for i in range(len(arrays)):
        assert arrays[i].shape == a.shape
        if len(shp) == 4:
            c[:,:,:,i] = arrays[i]
        elif len(shp) == 2:
            c[:,i] = arrays[i]
        else:
            raise         
    return c

def test_interleave_channel_arrays():
    a = numpy.array( (1,2,3,4,5,), dtype='uint16' )
    b = numpy.array( (6,7,8,9,10,), dtype='uint16' )
    # print (a)
    # print (b)
    c = interleave_channel_arrays( (a,b,) )
    # print (c)
    assert numpy.array_equal(c, numpy.array(
        [[ 1,  6],
         [ 2,  7],
         [ 3,  8],
         [ 4,  9],
         [ 5, 10]]))

def test_create_tiff():
    # https://pypi.python.org/pypi/tifffile
    fname = "E:/brunsc/projects/ktxtiff/octree_tip/default.0.tif"
    with TiffFile(fname) as tif:
        data1 = tif.asarray()
        # tifffile.imsave('test1.tif', data1)
    fname = "E:/brunsc/projects/ktxtiff/octree_tip/default.1.tif"
    with TiffFile(fname) as tif:
        data2 = tif.asarray()
        # tifffile.imsave('test2.tif', data2)
    # TODO unmixing test
    # compute channel 1/2 unmixing parameters
    # For lower end of mapping, just use lower quartile intensity (non-zero!)
    lower1 = numpy.percentile(data1[data1 != 0], 25)
    lower2 = numpy.percentile(data2[data2 != 0], 25)
    print (lower1, lower2)
    # For upper end of mapping, use voxels that are bright in BOTH channels
    m_a = numpy.median(data1[data1 != 0])
    m_b = numpy.median(data2[data2 != 0])
    s_a = numpy.std(data1[data1 != 0])
    s_b = numpy.std(data2[data2 != 0])
    upper1 = numpy.median(data1[(data1 > m_a + 4*s_a) & (data2 > m_b + 3*s_b)])
    upper2 = numpy.median(data2[(data1 > m_a + 4*s_a) & (data2 > m_b + 3*s_b)])
    print (upper1, upper2)
    # transform data2 to match data1
    scale = (upper1 - lower1) / (upper2 - lower2)
    offset = upper1 - upper2 * scale
    scale2 = (upper2 - lower2) / (upper1 - lower1)
    offset2 = upper2 - upper1 * scale2
    data2b = numpy.array(data2, dtype='float32')
    data2b *= scale
    data2b += offset
    data2b[data2 == 0] = 0
    data2b[data2b <= 0] = 0
    data2b = numpy.array(data2b, dtype=data1.dtype)
    # TODO ktx to tiff
    # Needs 1 or 3 channels for Fiji to load it OK
    data3 = numpy.zeros_like(data1)
    tissue = numpy.minimum(data1, data2)
    tissue_base = numpy.percentile(tissue[tissue != 0], 4) - 1
    tissue = numpy.array(tissue, dtype='float32') # so we can handle negative numbers
    print (tissue_base)
    tissue -= tissue_base
    tissue[tissue <= 0] = 0
    tissue = numpy.array(tissue, dtype=data1.dtype)
    #
    unmixed1 = numpy.array(data1, dtype='float32')
    unmixed1 -= data2b
    # unmixed1 += s_a # tweak background up to show more stuff
    unmixed1[unmixed1 <= 0] = 0
    unmixed1 = numpy.array(unmixed1, dtype=data1.dtype)
    #
    data1b = numpy.array(data1, dtype='float32')
    data1b *= scale2
    data1b += offset2
    data1b[data1 == 0] = 0
    data1b[data1b <= 0] = 0
    data1b = numpy.array(data1b, dtype=data1.dtype)
    unmixed2 = numpy.array(data2, dtype='float32')
    unmixed2 -= data1b
    # unmixed2 += s_b # tweak background up to show more stuff
    unmixed2[unmixed2 <= 0] = 0
    unmixed2 = numpy.array(unmixed2, dtype=data1.dtype)
    #
    print (tissue.shape)
    data123 = interleave_channel_arrays( (data2b, data1, data3) )
    # print (data123.shape)
    tifffile.imsave('test123.tif', data123)

def main():
    test_mipmap_dimension()
    test_downsample_xy()
    return
    "Interleave multiple single channel tiff files into a multi-channel KTX file"
    arrays = list()
    for arg in sys.argv[1:]:
        for fname in glob(arg):
            print (fname)
            with TiffFile(fname) as tif:
                print (len(tif.pages))
                data = tif.asarray()
                print (data.shape)
                arrays.append(data)
                # print (numpy.percentile(data[data != 0], [25, 99]))
    a = arrays[0]
    b = arrays[1]
    # TODO: generate linear unmixing parameters appropritate for both dim and bright sections
    # Use only non-zero locations for basic statistics
    m_a = numpy.median(a[a != 0])
    s_a = numpy.std(a[a != 0])
    m_b = numpy.median(b[b != 0])
    s_b = numpy.std(b[b != 0])
    # Statistic for locations where both channels are bright at the same location
    h_a = numpy.median(a[(a > m_a + 2*s_a) & (b > m_b + 2*s_b)])
    print (m_a, s_a, h_a)
    h_b = numpy.median(b[(a > m_a + 2*s_a) & (b > m_b + 2*s_b)])
    print (m_b, s_b, h_b)
    # Interleave two channels
    # http://stackoverflow.com/questions/5347065/interweaving-two-numpy-arrays
    c = numpy.empty( shape=(a.shape[0], a.shape[1], a.shape[2], len(arrays)), dtype=a.dtype)
    for i in range(len(arrays)):
        c[:,:,:,i] = arrays[i]
    print (c.shape)
    dt = c.dtype
    ktx = Ktx()
    kh = ktx.header
    if dt.byteorder == '<':
        kh.little_endian = True
    elif dt.byteorder == '=':
        kh.little_endian = sys.byteorder == 'little'
    else:
        raise # TODO
    print (dt.byteorder)
    print (kh.little_endian)
    if dt.kind == 'u':
        if dt.itemsize == 2:
            kh.gl_type = GL.GL_UNSIGNED_SHORT
        elif dt.itemsize == 1:
            kh.gl_type = GL.GL_UNSIGNED_BYTE
        else:
            raise # TODO
    else:
        raise # TODO
    #
    kh.gl_type_size = dt.itemsize
    #
    if c.shape[3] == 1:
        kh.gl_format = kh.gl_base_internal_format = GL.GL_RED
    elif c.shape[3] == 2:
        kh.gl_format = kh.gl_base_internal_format = GL.GL_RG
    elif c.shape[3] == 3:
        kh.gl_format = kh.gl_base_internal_format = GL.GL_RGB
    elif c.shape[3] == 4:
        kh.gl_format = kh.gl_base_internal_format = GL.GL_RGBA
    else:
        raise # TODO
    #
    if kh.gl_base_internal_format == GL.GL_RG and kh.gl_type == GL.GL_UNSIGNED_SHORT:
        kh.gl_internal_format = GL.GL_RG16UI
    else:
        raise # TODO
    #
    kh.pixel_width = c.shape[2]
    kh.pixel_height = c.shape[1]
    kh.pixel_depth = c.shape[0]
    kh.number_of_array_elements = 0
    kh.number_of_faces = 0
    kh.number_of_mipmap_levels = 1 # TODO zero for autogenerate?
    # TODO - key/value pairs for provenance
    ktx.image_data.mipmaps.clear()
    ktx.image_data.mipmaps.append(c.tostring())
    with io.open('test.ktx.lz4', 'wb') as ktx_out:
        temp = io.BytesIO()
        ktx.write_stream(temp)
        ktx_out.write(lz4.dumps(temp.getvalue()))
    
if __name__ == "__main__":
    test_create_tiff()
