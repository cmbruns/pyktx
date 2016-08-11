'''
Created on Aug 9, 2016

@author: brunsc
'''
import unittest
from io import BytesIO
import io
from glob import glob

from ktx import Ktx

class Test(unittest.TestCase):

    def testLoad1(self):
        img1 = Ktx()
        img1.read_filename("images/conftestimage_R11_EAC.ktx")
        # print("###")
        
    def testLoadAndSave(self):
        for fname in glob("images/*.ktx"):
            # print (fname)
            self._loadAndSaveOneImage(fname)
        
    def _loadAndSaveOneImage(self, fname):
        ktx1 = Ktx()
        ktx1.read_filename(fname)
        out = BytesIO()
        ktx1.write_stream(out)
        out.seek(0) # rewind before reading
        # print("####")
        Ktx().read_stream(out) # sanity check reading
        original = io.open(fname, 'rb').read() # load original file as binary in-memory blob
        # print(len(out.getvalue()))
        # print(len(original))
        assert out.getvalue() == original        
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()