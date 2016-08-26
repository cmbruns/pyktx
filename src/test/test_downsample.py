'''
Created on Aug 26, 2016

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

import unittest

import numpy

from ktx.util import _assort_subvoxels, mipmap_dimension

class Test(unittest.TestCase):

    def test_positive_control(self):
        assert True
        
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
    
    def testSubvoxelDownsampling(self):
        # Verify that each parent voxel appears in exactly one child subvoxel
        # over a range of dimension sizes, especially odd ones.
        # Simulate downsamping dimension of size 1
        a = _assort_subvoxels(numpy.array([101]), (1,))
        numpy.testing.assert_array_equal(a, 
                [[101]], # 
                "assort 1")
        # Simulate downsamping dimension of size 2
        a = _assort_subvoxels(numpy.array([101, 102]), (1,))
        numpy.testing.assert_array_equal(a, 
                [[101, 102]],
                "assort 2")
        # Simulate downsamping dimension of size 3
        a = _assort_subvoxels(numpy.array([101, 102, 103]), (1,))
        numpy.testing.assert_array_equal(a, 
                [[101, 102, 103]], #
                "assort 3")
        # Simulate downsamping dimension of size 4
        a = _assort_subvoxels(numpy.array([101, 102, 103, 104]), (2,))
        numpy.testing.assert_array_equal(a, 
                [[101, 102],
                 [103, 104]], 
                "assort 4")
        # Simulate downsamping dimension of size 5
        a = _assort_subvoxels(numpy.array(
                [101, 102, 103, 104, 105]), (2,))
        numpy.testing.assert_array_equal(a, 
                [[101, 102, 0], # Notice special zero value, to retain 1:1 mapping
                 [103, 104, 105]],
                "assort 5")
        # Simulate downsamping dimension of size 2, 5
        a = _assort_subvoxels(numpy.array(
                [[101, 102, 103, 104, 105],
                 [106, 107, 108, 109, 110]]), (2,2))
        numpy.testing.assert_array_equal(a, 
                [ [[101, 102, 0], [103, 104, 105]], 
                  [[106, 107, 0], [108, 109, 110]], ],
                "assort 5")
        # Simulate downsamping dimension of size 6
        a = _assort_subvoxels(numpy.array(
                [101, 102, 103, 104, 105, 106]), (3,))
        numpy.testing.assert_array_equal(a, 
                [[101, 102],
                 [103, 104],
                 [105, 106]],
                "assort 6")
        # Simulate downsamping dimension of size 7
        a = _assort_subvoxels(numpy.array(
                [101, 102, 103, 104, 105, 106, 107]), (3,))
        numpy.testing.assert_array_equal(a, 
                [[101, 102, 0],
                 [103, 104, 105],
                 [0, 106, 107]],
                "assort 7")
        # Simulate downsamping dimension of size 8
        a = _assort_subvoxels(numpy.array(
                [101, 102, 103, 104, 105, 106, 107, 108]), (4,))
        numpy.testing.assert_array_equal(a, 
                [[101, 102],
                 [103, 104],
                 [105, 106],
                 [107, 108]],
                "assort 8")
        # Simulate downsamping dimension of size 9
        a = _assort_subvoxels(numpy.array(
                [101, 102, 103, 104, 105, 106, 107, 108, 109]), (4,))
        numpy.testing.assert_array_equal(a, 
                [[101, 102, 0],
                 [103, 104, 0],
                 [105, 106, 107],
                 [0, 108, 109]],
                "assort 9")
        # Simulate downsamping dimension of size 11
        a = _assort_subvoxels(numpy.array(
                [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]), (5,))
        numpy.testing.assert_array_equal(a, 
                [[101, 102, 0],
                 [103, 104, 0],
                 [105, 106, 107], # middle column, only, gets three distinct subsamples
                 [0, 108, 109],
                 [0, 110, 111]],
                "assort 11")

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
