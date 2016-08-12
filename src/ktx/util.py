'''
package ktx.util

Created on Aug 11, 2016

@author: brunsc
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
    axis_offsets.append( (0,), )
    axis_step = list( (1,) )
    # Compute samples and stride for each dimension
    ndims = len(shape)
    for i in range(ndims):
        d = shape(i)
        factor = input_array.shape[i] / d
        axis_offsets.append( tuple(range(int(math.ceil(factor)))) ) # e.g. (0,1)
        axis_step.append( int(math.floor(factor)) )
    reduction_factor = 1
    for offset in axis_offsets:
        reduction_factor *= len(offset)
    scratch_shape = list(shape)
    scratch_shape.append(reduction_factor) # extra final dimension to hold subvoxel samples
    scratch = numpy.empty(shape=scratch_shape, dtype=input_array.dtype)
    # Compute strides into subvoxel list for each dimension
    pstrides = [1,] * ndims
    pstride = 1 # X dimension stride will be 1
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
            p_remainder -= p_remainder % stride
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
    return scratch

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

def create_mipmaps(mipmap0, filter_='arthur'):
    """
    Creates a sequence of mipmaps for a single-channel numpy image data array,
    Mipmap sizes follow OpenGL convention. 
    Filtering options:
      'mean' - average of parent voxels
      'max' - maximum of parent voxels
      'arthur' - second largest intensity among parent voxels (good for 
          preserving sparse, bright features, without too much noise)

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
        next_shape = list()
        axis_offsets = list() # Enumerates samples to combine in each direction
        axis_step = list()
        for i in range(ndims):
            d = mipmap_dimension(mipmap_level, biggest_shape[i])
            next_shape.append(d)
            factor = current_shape[i]/d
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
        next_shape = tuple(next_shape)
        # print(next_shape)
        # How many voxels combine into one at this mipmap level?
        reduction_factor = 1
        for offset in axis_offsets:
            reduction_factor *= len(offset)
        # print ("  ", reduction_factor, axis_offsets)
        # Re-assort data into a scratch buffer, so pixels to combine are in their own dimension
        scratch_shape = list(next_shape)
        scratch_shape.append(reduction_factor) # extra final dimension to hold subvoxel samples
        # print(scratch_shape)
        scratch = numpy.zeros(shape=scratch_shape, dtype=mipmap0.dtype)
        # Loop over each subvoxel to incorporate from parent (typically 2-27 subvoxel types total)
        # sz = len(axis_offsets[0])
        sy = len(axis_offsets[1])
        sx = len(axis_offsets[2])
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
                    scratch[scratch_key] = previous_mipmap[parent_key]
        # Generate mipmap
        # Combine those subvoxels into the final mipmap
        # Avoid zeros in mean/arthur computation
        useNan = True # nanpercentile is SOOO SLOWWWW
        if useNan:
            scratch = scratch.astype('float32') # 'float64' causes MemoryError?
            # Zero means no data, so set to "NaN" for filtering
            scratch[scratch==0] = numpy.nan
        if filter_ == 'mean':
            if useNan:
                mipmap = numpy.nanmean(scratch, axis=ndims) # Permit calculation to default to float dtype
            else:
                mipmap = numpy.mean(scratch, axis=ndims) # Permit calculation to default to float dtype                
        elif filter_ == 'max':
            if useNan:
                mipmap = numpy.nanmax(scratch, axis=ndims)
            else:
                mipmap = numpy.amax(scratch, axis=ndims)                
        elif filter_ == 'arthur': # second largest pixel value
            # percentile "82" yields second-largest value when number of elements is 7-12 (8 is canonical)
            if useNan:
                scratch = numpy.nan_to_num(scratch) # Put NaNs back to zeros
                scratch = numpy.sort(scratch)
                s0 = scratch[:,:,:,-1] # Largest element
                s1 = scratch[:,:,:,-2] # Second largest element
                s1[s1==0] = s0[s1==0] # Replace zeros with largest element
                mipmap = s1
                # mipmap = numpy.percentile(scratch, 82, axis=ndims, interpolation='higher')
                # Forget it; nanpercentile is crazy slow
                # mipmap = numpy.nanpercentile(scratch, 82, axis=ndims, interpolation='higher')
            else:
                mipmap = numpy.percentile(scratch, 82, axis=ndims, interpolation='higher')
        else:
            assert False # TODO: unknown filter
        mipmap = numpy.nan_to_num(mipmap) # Convert NaN to zero before writing
        mipmap = mipmap.astype(mipmap0.dtype) # Convert back to integer dtype AFTER calculation
        # print(mipmap.shape)
        current_shape = next_shape
        # print (mipmap_level, current_shape)
        previous_mipmap = mipmap
        mipmaps.append(mipmap)
    return mipmaps

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
        elif len(shp) == 2:
            c[:,i] = arrays[i]
        else:
            raise         
    return c
