#!/bin/env python

import os
import glob

maxjobs = 3000
jobcount = 0
for job in glob.glob("subtree*.sh"):
    jobcount += 1
    if jobcount > maxjobs:
        break
    # print (job)
    cmd = "qsub -cwd -j y -o %s.log -b y -l short=true -l h_rt=1500 -l d_rt=450 -V ./%s" % (job, job)
    print (cmd)
    os.system(cmd)

