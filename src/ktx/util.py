'''
package ktx.util

Created on Aug 11, 2016

@author: brunsc

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
'''

import math

import numpy

def mipmap_dimension(level, full):
    # Computes integer edge dimension for a mipmap of a particular level, based on the full sized (level zero) dimension
    return int(max(1, math.floor(full / 2**level)))

def _assort_subvoxels(input_array, shape):
    """
    Rearrange pixels before downsampling a volume, so subvoxels to be combined are in a new fast-moving dimension
    """
    axis_offsets = list() # Enumerates samples to combine in each direction
    axis_step = list() # Sampling stride in each dimension
    # Compute samples and stride for each dimension
    ndims = len(shape)
    for i in range(ndims):
        d = shape[i]
        factor = input_array.shape[i] / d
        axis_offsets.append( tuple(range(int(math.ceil(factor)))) ) # e.g. (0,1)
        axis_step.append( int(math.floor(factor)) ) # e.g. "2"
    reduction_factor = 1
    for offset in axis_offsets:
        reduction_factor *= len(offset)
    #
    scratch_shape = list(shape)
    scratch_shape.append(reduction_factor) # extra final dimension to hold subvoxel samples
    scratch = numpy.zeros(shape=scratch_shape, dtype=input_array.dtype)
    # Compute strides into subvoxel list for each dimension
    pstrides = [1,] * ndims
    pstride = 1 # X (fastest) dimension stride will be 1
    for i in reversed(range(ndims)):
        pstrides[i] = pstride
        pstride *= len(axis_offsets[i])
    # loop over each subvoxel comprising one voxel, and populate new array
    for p in range(reduction_factor): 
        # Compute subvoxel offset in each dimension
        axis_start = list()
        p_remainder = p
        for i in range(ndims): # loop over each dimension axis
            stride = pstrides[i] # distance between adjacent subvoxels in this dimension
            start = p_remainder // stride # index of subvoxel in this dimension
            axis_start.append(start)
            p_remainder -= start * stride
        # Generate slice to extract this subvoxel from the parent mipmap
        parent_key = list()
        for i in range(ndims):
            start = axis_start[i] # Initial offset along axis
            end = axis_start[i] + scratch_shape[i] * axis_step[i] # Final offset along axis, +1
            step = axis_step[i] # Stride along axis
            slice_ = slice(start, end, step) # partial key for our fancy data slurp, below
            parent_key.append(slice_)
        subvoxel_index = p
        scratch_key = [slice(None), ] * ndims + [subvoxel_index,] # e.g. [:,:,:,0]
        # Slurp every instance of this subvoxel into the scratch array
        scratch[scratch_key] = input_array[parent_key]
    # Zero certain subvoxels when downsampling odd numbered parent dimensions,
    # So each parent voxel contributes to exactly one child subvoxel
    # Reshape to make voxel zeroing easier
    shape1 = tuple(scratch.shape) # shape with flattened subvoxels
    subvoxel_shape = tuple([len(offset) for offset in axis_offsets])
    shape2 = tuple(shape1[0:-1] + subvoxel_shape)
    scratch.shape = shape2
    for i in range(ndims):
        input_len = input_array.shape[i]
        output_len = shape[i]
        if input_len != 2 * output_len + 1:
            continue # Not the case we are looking for
        if output_len < 2:
            continue # No zeroing needed for input sizes 1 and 3
        # Only one column gets to keep all three of its subsampled values
        pivot_column = output_len // 2
        # 
        # For multidimensional subvoxels, we want to zero multiple subvoxels for this dimension
        stride = pstrides[i]
        #
        # print (input_len, output_len, pivot_column)
        # Clear third pixel of leftmost columns
        left_key = [slice(None),] * i # Everything from earlier dimensions
        left_key.append(slice(0, pivot_column)) # Left half of this dimension
        left_key.extend([slice(None),] * (len(input_array.shape) - i - 1)) # later dimensions
        # Subvoxel portion of key below
        left_key.extend([slice(None),] * i) # Everything from earlier dimensions
        left_key.append(slice(2, 3, 1)) # Clear the third subvoxel in this dimension
        left_key.extend([slice(None),] * (len(input_array.shape) - i - 1))
        scratch[left_key] = 0
        # print (left_key)
        # Clear first pixel of rightmost columns
        right_key = [slice(None),] * i # Everything from earlier dimensions
        right_key.append(slice(pivot_column + 1, output_len)) # Right half of this dimension
        right_key.extend([slice(None),] * (len(input_array.shape) - i - 1)) # later dimensions
        # Subvoxel portion of key below
        right_key.extend([slice(None),] * i) # Everything from earlier dimensions
        right_key.append(slice(0, 1, 1)) # Clear the first subvoxel in this dimension
        right_key.extend([slice(None),] * (len(input_array.shape) - i - 1))
        scratch[right_key] = 0
    scratch.shape = shape1
    return scratch

def _filter_assorted_array(assorted_array, filter_='mean'):
    """
    Apply box-like downsample filter to specially prearranged image array.
    The input assorted_array must have the following properties:
      * the final fastest-changing dimension contains all the parent voxels
        that contribute to the new final downsampled voxel
      * there is only a single color channel
      * (see _assort_subvoxels)
    Filtering options:
      'mean' - average intensity of parent voxels
      'max' - maximum intensity of parent voxels
      'arthur' - second largest intensity among parent voxels (good for 
          preserving sparse, bright features, without too much noise)
    """
    # Combine those subvoxels into the final mipmap
    # Avoid zeros in mean/arthur computation
    original_dtype = assorted_array.dtype
    ndims = len(assorted_array.shape) - 1 # "1" Because it has an extra dimension for the subvoxels
    useNan = True # nanpercentile is SOOO SLOWWWW
    if useNan and filter_ != 'arthur':
        assorted_array = assorted_array.astype('float32') # 'float64' causes MemoryError?
        # Zero means no data, so set to "NaN" for filtering
        assorted_array[assorted_array==0] = numpy.nan
    if filter_ == 'mean':
        if useNan:
            mipmap = numpy.nanmean(assorted_array, axis=ndims) # Permit calculation to default to float dtype
        else:
            mipmap = numpy.mean(assorted_array, axis=ndims) # Permit calculation to default to float dtype                
    elif filter_ == 'max':
        if useNan:
            mipmap = numpy.nanmax(assorted_array, axis=ndims)
        else:
            mipmap = numpy.amax(assorted_array, axis=ndims)                
    elif filter_ == 'arthur': # second largest pixel value
        assorted_array = numpy.sort(assorted_array) # sort intensities of subvoxels along final dimension
        brightest_key = [slice(None),]*ndims + [-1,]
        second_brightest_key = [slice(None),]*ndims + [-2,]
        s0 = assorted_array[brightest_key] # Largest intensity per voxel
        s1 = assorted_array[second_brightest_key] # Second largest intensity per voxel
        s1[s1==0] = s0[s1==0] # Replace zeros with largest element, in case second largest is zero/no-data
        mipmap = s1
        # percentile "82" yields second-largest value when number of elements is 7-12 (8 is canonical)
        # mipmap = numpy.percentile(assorted_array, 82, axis=ndims, interpolation='higher')
        # Forget it; nanpercentile is crazy slow
        # mipmap = numpy.nanpercentile(assorted_array, 82, axis=ndims, interpolation='higher')
    else:
        raise Exception("Unknown downsampling filter %s" % filter_)
    mipmap = numpy.nan_to_num(mipmap) # Convert NaN to zero before writing
    mipmap = mipmap.astype(original_dtype) # Convert back to integer dtype AFTER calculation
    return mipmap

def downsample_array_xy(array, filter_='arthur'):
    """
    Downsample in X and Y directions, using second largest non-zero intensity.
    """
    shape = list(tuple(array.shape))
    for i in (1,2): # Y and X dimensions
        shape[i] = mipmap_dimension(1, shape[i])
    scratch = _assort_subvoxels(array, shape)
    downsampled = _filter_assorted_array(scratch, filter_)
    return downsampled

def create_mipmaps(mipmap0, filter_='arthur'):
    """
    Creates a sequence of mipmaps for a single-channel numpy image data array,
    Mipmap sizes follow OpenGL convention. 
    """
    mipmaps = list()
    mipmaps.append(mipmap0) # First mipmap is the input impage
    biggest_shape = tuple(mipmap0.shape)
    ndims = len(biggest_shape)
    smallest_shape = tuple([1] * ndims) # Final mipmap will have all dimensions equal to "1"
    # print (biggest_shape, smallest_shape)
    current_shape = tuple(biggest_shape)
    mipmap_level = 0
    # print (mipmap_level, current_shape)
    previous_mipmap = mipmap0
    while current_shape != smallest_shape:
        mipmap_level += 1
        next_shape = tuple([mipmap_dimension(mipmap_level, biggest_shape[i]) for i in range(ndims)]) 
        scratch = _assort_subvoxels(previous_mipmap, next_shape)
        mipmap = _filter_assorted_array(scratch, filter_)
        # print(mipmap.shape)
        current_shape = next_shape
        # print (mipmap_level, current_shape)
        previous_mipmap = mipmap
        mipmaps.append(mipmap)
    return mipmaps

def mipmap_shapes(root_shape):
    """
    Creates a sequence of mipmap shapes.
    Mipmap sizes follow OpenGL convention. 
    """
    mipmap_shapes = []
    biggest_shape = tuple(root_shape)
    mipmap_shapes.append(biggest_shape) # First mipmap is the input impage
    ndims = len(biggest_shape)
    smallest_shape = tuple([1] * ndims) # Final mipmap will have all dimensions equal to "1"
    current_shape = tuple(biggest_shape)
    mipmap_level = 0
    while current_shape != smallest_shape:
        mipmap_level += 1
        current_shape = tuple([mipmap_dimension(mipmap_level, biggest_shape[i]) for i in range(ndims)]) 
        mipmap_shapes.append(current_shape)
    return mipmap_shapes

def interleave_channel_arrays(arrays):
    "Combine multiple single channel stacks into one multi-channel stack"
    a = arrays[0]
    shp = list(a.shape)
    # print (shp)
    # print (len(shp))
    shp.append(len(arrays))
    # print(shp)
    # print (a.dtype)
    c = numpy.empty( shape=shp, dtype=a.dtype)
    for i in range(len(arrays)):
        assert arrays[i].shape == a.shape
        if len(shp) == 4:
            c[:,:,:,i] = arrays[i]
        elif len(shp) == 3:
            c[:,:,i] = arrays[i]
        elif len(shp) == 2:
            c[:,i] = arrays[i]
        else:
            raise         
    return c
