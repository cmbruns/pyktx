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
import time
from ktx.util import create_mipmaps, mipmap_dimension, interleave_channel_arrays

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
      * texture coordinate bounds for display (because there might be padding...)
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
    

def test_downsample_xy(filter_='arthur'):
    fname = "E:/brunsc/projects/ktxtiff/octree_tip/default.0.tif"
    with TiffFile(fname) as tif:
        data = tif.asarray()
    t0 = time.time()
    downsampled = downsample_array_xy(data, filter_=filter_)
    t1 = time.time()
    print (t1-t0, " seconds elapsed time to downsample volume in XY")
    tifffile.imsave("downsampled.tif", downsampled)
    t2 = time.time()
    print (t2-t1, " seconds elapsed time to save downsampled volume in tiff format to disk")

def downsample_array_xy(array, filter_='arthur'):
    """
    Downsample in X and Y directions, using second largest non-zero intensity.
    TODO: reduce duplicated code compared to create_mipmaps
    """
    # Initialize first dimension, Z, with no downsampling
    shape = list( (array.shape[0],) )
    axis_offsets = list() # Enumerates samples to combine in each direction
    axis_offsets.append( (0,), )
    axis_step = list( (1,) )
    # Downsample X and Y dimensions
    for i in range(1,3):
        d = mipmap_dimension(1, array.shape[i])
        shape.append(d)
        factor = array.shape[i] / d
        if factor > 2.0:
            axis_offsets.append((0, 1, 2,),) # combine three samples along odd-dimensioned axes
            axis_step.append(2)
        elif factor == 2.0:
            axis_offsets.append((0, 1,),) # combine two samples along even-dimensioned axes
            axis_step.append(2)
        elif factor == 1.0:
            axis_offsets.append((0,),) # single sample along one-length dimensions
            axis_step.append(1)
        else:
            assert False # should not happen
    shape = tuple(shape)
    reduction_factor = 1
    for offset in axis_offsets:
        reduction_factor *= len(offset)
    scratch_shape = list(shape)
    scratch_shape.append(reduction_factor) # extra final dimension to hold subvoxel samples
    scratch = numpy.empty(shape=scratch_shape, dtype=array.dtype)
    sy = len(axis_offsets[1])
    sx = len(axis_offsets[2])
    ndims = 3 # TODO: don't hardcode this...
    for z in axis_offsets[0]:
        for y in axis_offsets[1]:
            for x in axis_offsets[2]:
                axis_start = (z, y, x)
                # Generate slice to extract this subvoxel from the parent mipmap
                parent_key = list()
                for i in range(ndims):
                    start = axis_start[i] # Initial offset along axis
                    end = axis_start[i] + scratch_shape[i] * axis_step[i] # Final offset along axis, +1
                    step = axis_step[i] # Stride along axis
                    slice_ = slice(start, end, step) # partial key for our fancy data slurp, below
                    parent_key.append(slice_)
                subvoxel_index = x + y*sx + z*sx*sy
                scratch_key = [slice(None), ] * ndims + [subvoxel_index,] # e.g. [:,:,:,0]
                # Slurp every instance of this subvoxel into the scratch array
                scratch[scratch_key] = array[parent_key]
    # Generate mipmap
    # Combine those subvoxels into the final mipmap
    # Avoid zeros in mean/arthur computation
    useNan = False
    if useNan:
        scratch = scratch.astype('float32') # 'float64' causes MemoryError?
        # Zero means no data, so set to "NaN" for filtering
        scratch[scratch==0] = numpy.nan
    if filter_ == 'mean':
        if useNan:
            downsampled = numpy.nanmean(scratch, axis=ndims) # Permit calculation to default to float dtype
        else:
            downsampled = numpy.mean(scratch, axis=ndims) # Permit calculation to default to float dtype                
    elif filter_ == 'max':
        if useNan:
            downsampled = numpy.nanmax(scratch, axis=ndims)
        else:
            downsampled = numpy.amax(scratch, axis=ndims)                
    elif filter_ == 'arthur': # second largest pixel value
        # percentile "82" yields second-largest value when number of elements is 7-12 (8 is canonical)
        if useNan:
            downsampled = numpy.nanpercentile(scratch, 82, axis=ndims, interpolation='higher')
        else:
            downsampled = numpy.percentile(scratch, 82, axis=ndims, interpolation='higher')
    else:
        assert False # TODO: unknown filter
    downsampled = numpy.nan_to_num(downsampled) # Convert NaN to zero before writing
    downsampled = downsampled.astype(array.dtype) # Convert back to integer dtype AFTER calculation
    return downsampled

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

def test_create_mipmaps(filter_='arthur'):
    fname = "E:/brunsc/projects/ktxtiff/octree_tip/default.0.tif"
    with TiffFile(fname) as tif:
        data = tif.asarray()
    data = downsample_array_xy(data, filter_=filter_)
    t0 = time.time()
    mipmaps = create_mipmaps(data, filter_=filter_)
    t1 = time.time()
    print (t1-t0, " seconds elapsed time to compute mipmaps")
    for i in range(len(mipmaps)):
        tifffile.imsave("test_mipmap%02d.tif" % i, mipmaps[i])
    t2 = time.time()
    print (t2-t1, " seconds elapsed time to save mipmaps in tiff format to disk")

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
    lower1 = numpy.percentile(data1[data1 != 0], 40)
    lower2 = numpy.percentile(data2[data2 != 0], 40)
    print (lower1, lower2)
    # For upper end of mapping, use voxels that are bright in BOTH channels
    m_a = numpy.median(data1[data1 != 0])
    m_b = numpy.median(data2[data2 != 0])
    s_a = numpy.std(data1[data1 != 0])
    s_b = numpy.std(data2[data2 != 0])
    upper1 = numpy.median(data1[(data1 > m_a + 2*s_a) & (data2 > m_b + 2*s_b)])
    upper2 = numpy.median(data2[(data1 > m_a + 2*s_a) & (data2 > m_b + 2*s_b)])
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
    # data3 = numpy.zeros_like(data1)
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
    data123 = interleave_channel_arrays( (data2, data1b, unmixed2) )
    # print (data123.shape)
    tifffile.imsave('test123.tif', data123)

def ktx_from_tiff_channel_files(channel_tiff_names, mipmap_filter='max', downsample_xy=True, downsample_intensity=False):
    """
    Load multiple single-channel tiff files, and create a multichannel Ktx object.
    Mipmap voxel filtering options:
      None - no mipmaps will be generated
      'mean' - average of parent voxels
      'max' - maximum of parent voxels
      'arthur' - second largest intensity among parent voxels (good for 
          preserving sparse, bright features, without too much noise)
    """
    channels = list()
    for fname in channel_tiff_names:
        with TiffFile(fname) as tif:
            arr = tif.asarray()
            if downsample_xy:
                arr = downsample_array_xy(arr, mipmap_filter)
            channels.append(arr)
    # TODO: remove this kludge to get a decent TIFF when there are only 2 channels
    if len(channels) == 2:
        channels.append(numpy.zeros_like(channels[0]))
    combined = interleave_channel_arrays(channels)
    ktx = Ktx.from_ndarray(combined, mipmap_filter=mipmap_filter)
    # Additional metadata, from personal knowledge
    kv = ktx.header.key_value_metadata
    kv[b'distance_units'] = b'micrometers\x00'
    xform = numpy.array([ # TODO: use correct values in matrix...
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],], dtype='float32')
    kv[b'xyz_from_texcoord_xform'] = xform.tostring()
    # Write LZ4-compressed file
    with io.open('test.ktx.lz4', 'wb') as ktx_out:
        temp = io.BytesIO()
        ktx.write_stream(temp)
        compressed = lz4.dumps(temp.getvalue())
        ktx_out.write(compressed)
    # ktx.populate_from_ndarray(combined) 
    # TODO: remove tiff writing after debugging
    # Testing
    tifffile.imsave('combined.tif', combined)

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
    ktx_from_tiff_channel_files(
            ("E:/brunsc/projects/ktxtiff/octree_tip/default.1.tif",
            "E:/brunsc/projects/ktxtiff/octree_tip/default.0.tif",
            ), )
