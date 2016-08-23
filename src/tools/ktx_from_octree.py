#!/bin/env python

"""
Copyright (c) 2016 Christopher M. Bruns

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# built-in python modules
import io
import time
from glob import glob
import os
import math
import datetime

# third party python modules
from OpenGL import GL
from tifffile import TiffFile
import tifffile
import numpy

# local python modules
import ktx
from ktx.util import create_mipmaps, mipmap_dimension, interleave_channel_arrays, downsample_array_xy

"""
TODO: For converting rendered octree blocks, include the following precomputed:
  * all mipmap levels DONE
  * optional intensity downsampling, with affine reestimation parameters DONE
  * optional spatial downsampling (XY DONE)
  * other metadata:
      * distance units e.g. "micrometers", for all the transforms below DONE
      * transform from texture coordinates to Cartesian reference space TODO: box corners
      * Optional transform from texture coordinates to Allen reference space TODO:
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

def ktx_from_mouselight_octree_folder(input_folder_name,
                              output_folder_name,
                              max_levels=1, # '0' means 'all'
                              mipmap_filter='max', 
                              downsample_xy=True, 
                              downsample_intensity=False):
    # Parse geometry data from top level transform.txt file in octree folder
    metadata = dict()
    with io.open(os.path.join(input_folder_name, "transform.txt"), 'r') as transform_file:
        for line in transform_file:
            fields = line.split(": ")
            if len(fields) != 2:
                continue
            metadata[fields[0].strip()] = fields[1].strip()
    # Get original tiff file dimensions, to help compute geometry correctly
    with tifffile.TiffFile(os.path.join(input_folder_name, "default.0.tif")) as tif:
        original_tiff_dimensions = tif.asarray().shape
    # for k, v in metadata.items():
    #     print (k, v)
    if max_levels == 0:
        max_levels = int(metadata["nl"])
    assert max_levels > 0
    folder = input_folder_name
    for level in range(max_levels):
        tiffs = glob(os.path.join(folder, "default.*.tif"))
        ktx_obj = ktx_from_tiff_channel_files(tiffs, mipmap_filter, downsample_xy, downsample_intensity)
        # Populate custom block metadata
        kh = ktx_obj.header
        # kv = ktx_obj.header.key_value_metadata
        # kv[b'distance_units'] = b'micrometers'
        kh["distance_units"] = "micrometers"
        umFromNm = 1.0/1000.0
        # Origin of volume (corner of corner voxel)
        ox = umFromNm*float(metadata['ox'])
        oy = umFromNm*float(metadata['oy'])
        oz = umFromNm*float(metadata['oz'])
        # Size of entire volume
        # Use original dimensions, to account for downsampling...
        sx = umFromNm * original_tiff_dimensions[2] * float(metadata['sx']) 
        sy = umFromNm * original_tiff_dimensions[1] * float(metadata['sy'])
        sz = umFromNm * original_tiff_dimensions[0] * float(metadata['sz'])
        xform = numpy.array([
                [sx, 0, 0, ox],
                [0, sy, 0, oy],
                [0, 0, sz, oz],
                [0, 0, 0, 1],], dtype='float64')
        # print(xform)
        kh["xyz_from_texcoord_xform"] = xform
        # print (kh["xyz_from_texcoord_xform"])
        #
        center = numpy.array( (ox + 0.5*sx, oy + 0.5*sy, oz + 0.5*sz,), )
        radius = math.sqrt(sx*sx + sy*sy + sz*sz)/16.0
        kh['bounding_sphere_center'] = center
        kh['bounding_sphere_radius'] = radius
        # Nominal resolution
        resX = sx / ktx_obj.header.pixel_width
        resY = sy / ktx_obj.header.pixel_height
        resZ = sz / ktx_obj.header.pixel_depth
        rms = math.sqrt(numpy.mean(numpy.square([resX, resY, resZ],)))
        kh['nominal_resolution'] = rms
        # print (kh['nominal_resolution'])
        # Specimen ID
        kh['specimen_id'] = os.path.split(input_folder_name)[-1]
        # print (kh['specimen_id'])
        # TODO: octree block ID
        # Relation to parent tile/block
        kh['mipmap_filter'] = mipmap_filter
        relations = list()
        if downsample_xy:
            relations.append("downsampled 2X in X & Y")
        if downsample_intensity:
            relations.append("rescaled intensity to 8 bits")
        if len(relations) == 0:
            relations.append("unchanged")
        kh['relation_to_parent'] = ";".join(relations)
        # print (kh['relation_to_parent'])
        kh['multiscale_level_id'] = level
        kh['multiscale_total_levels'] = metadata['nl']
        # TODO: Per channel statistics
        kh['ktx_file_creation_date'] = datetime.datetime.now()
        # print (kh['ktx_file_creation_date'])
        import __main__ #@UnresolvedImport
        kh['ktx_file_creation_program'] = __main__.__file__
        # print (kh['ktx_file_creation_program'])
        kh['pyktx_version'] = ktx.__version__
        # print (kh['ktx_package_version'])
        # TODO: Texture coordinate bounds for display
        # Write LZ4-compressed KTX file
        t1 = time.time()
        with io.open('test.ktx', 'wb') as ktx_out:
            temp = io.BytesIO()
            ktx_obj.write_stream(temp)
            ktx_out.write(temp.getvalue())        # Create tiff file for sanity check testing
        t2 = time.time()
        print ("Creating uncompressed ktx file took %.3f seconds" % (t2 - t1))

def arrays_from_tiff_channel_files(channel_tiff_names):
    channels = list()
    for fname in channel_tiff_names:
        with TiffFile(fname) as tif:
            arr = tif.asarray()
            channels.append(arr)
    return channels

def report_array_stats(channels, description):
    channel_mins = [numpy.min(c[c!=0]) for c in channels]
    print ("Minimum non-zero %s channel intensities = %s" % (description, channel_mins,))
    channel_medians = [numpy.median(c[c!=0]) for c in channels]
    print ("Median non-zero %s channel intensities = %s" % (description, channel_medians,))
    channel_maxes = [numpy.max(c) for c in channels]
    print ("Maximum %s channel intensities = %s" % (description, channel_maxes,))
    channel_stddevs = [numpy.std(c[c!=0]) for c in channels]
    print ("Standard deviation of non-zero %s channel intensities = %s" % (description, channel_stddevs,))

def ktx_from_tiff_channel_files(channel_tiff_names, mipmap_filter='max', downsample_xy=True, downsample_intensity=False):
    """
    Load multiple single-channel tiff files, and create a multichannel Ktx object.
    """
    t0 = time.time()
    channels = arrays_from_tiff_channel_files(channel_tiff_names)
    original_channels = channels
    t1 = time.time()
    print ("loading tiff files into RAM took %.3f seconds" % (t1 - t0))
    report_array_stats(original_channels, "original TIFF")
    if downsample_xy:
        t0 = time.time()
        channels = [downsample_array_xy(c, mipmap_filter) for c in original_channels]
        t1 = time.time()
        print ("downsampling tiff files in X and Y using %s filter took %.3f seconds" % (mipmap_filter, t1 - t0))
    t0 = time.time()
    if downsample_intensity:
        new_channels = list()
        channel_transforms = list()
        for channel in channels:
            min_ = numpy.min(channel[channel != 0])
            max_ = numpy.max(channel[channel != 0])
            scale = 1.0
            offset = min_ - 1
            if max_ - min_ > 255: # need a lossy contraction of intensities
                # Discard dimmest 2% of intensities
                min_ = numpy.percentile(channel[channel != 0], 2)
                median = numpy.median(channel[channel != 0])
                max_ = numpy.max(channel[channel != 0])
                # Discard intensities above 90% of max
                max_ = median + 0.90 * (max_ - median)
                # print(min_, median, max_)
                scale = (max_ - min_) / 255.0
                offset = min_ - 1
            if channel.dtype.itemsize == 2:
                c = numpy.array(channel, dtype='float64')
                c -= offset
                c /= scale
                c[c<0] = 0
                c[c>255] = 255
                if channel.dtype == numpy.uint16:
                    dt = numpy.uint8
                else:
                    raise # TODO: more cases
                c = numpy.array(c, dtype=dt)
                new_channels.append(c)
                channel_transforms.append( tuple([scale, offset]) )
            else:
                raise # TODO:
        channels = new_channels
        t1 = time.time()
        print ("downsampling tiff intensities took %.3f seconds" % (t1 - t0))
    report_array_stats(channels, "final output")
    t0 = time.time()
    combined = interleave_channel_arrays(channels)
    t1 = time.time()
    print ("interleaving color channels took %.3f seconds" % (t1 - t0))
    ktx_obj = ktx.Ktx.from_ndarray(combined, mipmap_filter=mipmap_filter)
    # Include metadata for reconstructing original intensities
    if downsample_intensity:
        c = 0
        for ct in channel_transforms:
            ktx_obj.header['intensity_transform_%d'%c] = ct
            c += 1
    t2 = time.time()
    print ("computing mipmaps took %.3f seconds" % (t2 - t1))
    # For debugging, reconstruct data and compute mean squared error
    reconstructed = list(channels)
    if downsample_intensity:
        # Reverse intensity scaling transform
        reconstructed = [numpy.array(c, dtype='float64') for c in reconstructed]
        # Rescale
        reconstructed = [reconstructed[i] * channel_transforms[i][0] for i in range(len(reconstructed))]
        # Offset
        reconstructed = [reconstructed[i] + channel_transforms[i][1] for i in range(len(reconstructed))]
        # Restore zero values
        for i in range(len(reconstructed)):
            c0 = channels[i]
            c1 = reconstructed[i]
            c1[c0 == 0] = 0 # zero means "no data"
            c1[c1 < 0] = 1 # very small non-zero values become ones
    if downsample_xy:
        # TODO: Upsample to original size. This could be hard,
        #  but I need this to be able to compute RMS error
        downsampled = reconstructed
        down_shape = downsampled[0].shape
        down_key = [slice(None),] * len(down_shape) # Select everything every time
        upsampled = [numpy.zeros_like(c) for c in original_channels]
        up_shape = upsampled[0].shape
        # print (up_shape)
        # TODO: If shape dimension is ODD, we should fill right half differently...
        for c in range(len(original_channels)):
            for dy in (0,1):
                for dx in (0,1):
                    up_key = [slice(None), # all Z
                              slice(dy, up_shape[1], 2), # alternate y values
                              slice(dx, up_shape[2], 2), # alternate x values
                              ] + ([slice(None)] * (len(up_shape) - 3))
                    u = upsampled[c]
                    d = downsampled[c]
                    u[up_key] = d[down_key]
        # print (down_key, up_key)
        reconstructed = upsampled
    report_array_stats(reconstructed, "reconstructed")
    err = list()
    for c in range(len(original_channels)):
        orig = original_channels[c]
        recon = reconstructed[c]
        diff = recon - orig # deviation
        diff = diff*diff # squared
        mean = numpy.mean(diff[orig != 0])
        err.append(math.sqrt(mean))
    print ("root-mean-square non-zero reconstructed channel intensity errors = %s" % (err,))
    # TODO mean squared error
    return ktx_obj
 
if __name__ == "__main__":
    # TODO: Parse command line arguments
    ktx_from_mouselight_octree_folder(
            input_folder_name='//fxt/nobackup2/mouselight/2015-06-19-johan-full', 
            output_folder_name='',
            mipmap_filter='arthur', 
            downsample_xy=True,
            downsample_intensity=True,
            max_levels=1)

