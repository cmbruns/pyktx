#!/bin/env python

import os
import stat
import glob
import subprocess
import time

start_time = time.time()
maxjobs = 5000
# maxjobs = 3000
first_job = 0
jobcount = 0
skipped_jobs = 0
max_concurrent_jobs = 250
qstat_freq = 25
qstat_countdown = 0
os.chdir('jobscripts_150619_prob_octant12_FC_knninit')
all_jobs = glob.glob("subtree*.sh")
for job in all_jobs:
    if first_job > skipped_jobs:
        skipped_jobs += 1
        continue
    jobcount += 1
    if jobcount > maxjobs:
        break
    # Make sure script is executable
    permissions = os.stat(job).st_mode
    desired_permissions = permissions | stat.S_IEXEC
    if permissions != desired_permissions:
        os.chmod(job, desired_permissions)
    # Don't submit so many jobs at once
    if qstat_countdown <= 0:
        print("%d jobs submitted so far, out of %d total" % (jobcount, len(all_jobs)))
        job_list = subprocess.check_output(['qstat',])
        running_count = len(job_list.splitlines()) - 2
        while running_count > (max_concurrent_jobs - qstat_freq):
            sleep_interval = 10
            print("%d jobs are currently running. Waiting %d seconds before checking again..." % (running_count, sleep_interval) )
            time.sleep(10)
            job_list = subprocess.check_output(['qstat', ])
            running_count = len(job_list.splitlines()) - 2
        qstat_countdown = qstat_freq
    qstat_countdown -= 1
    # print (job)
    cmd = "qsub -cwd -j y -b y -o %s.log -l h_rt=3600 -l d_rt=750 -V ./%s" % (job, job)
    print (cmd)
    os.system(cmd)
elapsed_time = time.time() - start_time
print("Submitted %d jobs in %d seconds" % (elapsed_time, len(all_jobs)))
