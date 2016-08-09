'''
Created on Aug 9, 2016

@author: brunsc
'''
import unittest

from ktx import Ktx

class Test(unittest.TestCase):

    def testLoad1(self):
        img1 = Ktx()
        img1.load_filename("conftestimage_R11_EAC.ktx")


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
