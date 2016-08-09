'''
Created on Aug 9, 2016

@author: brunsc
'''
import unittest
from io import BytesIO
import io

from ktx import Ktx

class Test(unittest.TestCase):

    def testLoad1(self):
        img1 = Ktx()
        img1.read_filename("conftestimage_R11_EAC.ktx")
        print("###")
        
    def testLoadAndSave(self):
        fname = "conftestimage_R11_EAC.ktx"
        ktx1 = Ktx()
        ktx1.read_filename(fname)
        out = BytesIO()
        ktx1.write_stream(out)
        out.seek(0)
        print("####")
        Ktx().read_stream(out)
        original = io.open(fname, 'rb').read()
        assert out.getvalue() == original
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
