#!/misc/local/python3/bin/python3.5

import os

script_base = """\
#!/bin/sh
export KTXSRC='/home/brunsc/git/pyktx/src'
export PYTHONPATH="$KTXSRC"
echo $PYTHONPATH
/misc/local/python3/bin/python3.5 $KTXSRC/tools/convert_subtree.py "%s" "%s" "%s" %d
"""

input_root = "/nobackup2/mouselight/2016-04-04b"
output_root = "/nobackup2/mouselight/brunsc/ktxtest"
num_levels = 3

def recurse_octree(folder, level):
    if not os.path.exists(folder):
        return
    # if level > 4:
    #     return
    if level % num_levels == 2: # just levels 2 and 5 (we'll fill in level 1 manually)
        print (folder)
        if level == 1:
            subtree0 = []
        else:
            subtree0 = folder.split('/')[-(level-1):]
        subtree = "/".join(subtree0)
        print (subtree)
        script = script_base % (input_root, output_root, subtree, num_levels)
        # print (script)
        script_name = "subtree%s.sh" % "".join(subtree0)
        print (script_name)
        if True:
            f = open('jobscripts/%s' % script_name, 'w')
            f.write(script)
            f.close()
    # return
    for i0 in range(8):
        subfolder = folder + '/' + str(i0+1)
        recurse_octree(subfolder, level + 1)

recurse_octree(input_root, 1)

