#!/bin/env python

import os
import glob

maxjobs = 4200
# maxjobs = 3000
first_job = 0
jobcount = 0
skipped_jobs = 0
os.chdir('jobscripts_150619_prob_octant12')
for job in glob.glob("subtree*.sh"):
    if first_job > skipped_jobs:
        skipped_jobs += 1
        continue
    jobcount += 1
    if jobcount > maxjobs:
        break
    # print (job)
    cmd = "qsub -cwd -j y -b y -o %s.log -l h_rt=3600 -l d_rt=750 -V ./%s" % (job, job)
    print (cmd)
    os.system(cmd)

