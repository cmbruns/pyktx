#!/bin/env python

import os

# Keep rerunning incremental render until it succeeds
for i in range(20):
    cmd = "py -3 ktx_from_rendered_tiff.py >> big_render6.log"
    print (cmd)
    os.system(cmd)

