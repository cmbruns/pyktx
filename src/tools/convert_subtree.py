#!/bin/env python3
'''
Created on Sep 30, 2016

@author: brunsc
'''

import os
import sys
import time

from ktx.octree.ktx_from_rendered_tiff import RenderedMouseLightOctree


def convert_subtree(            
            input_root_folder,
            output_root_folder,
            subtree_tip=[],
            subtree_level_count=1,
            downsample_intensity=True, 
            downsample_xy=True,
            overwrite=False):
    o = RenderedMouseLightOctree(
            input_folder=os.path.abspath(input_root_folder), 
            downsample_intensity=downsample_intensity,
            downsample_xy=downsample_xy)
    tip = "/".join([str(n) for n in subtree_tip])
    tip_folder = o.input_folder + "/" + tip
    print ( tip_folder )
    specimen = os.path.split(input_root_folder)[1]
    for block in o.iter_blocks(folder=tip_folder, max_level=subtree_level_count-1):
        # print(block.octree_path)
        file_name = 'block_' + specimen
        if downsample_intensity:
            file_name += '_8'
        if downsample_xy:
            file_name += '_xy'
        opath1 = "".join([str(n) for n in block.octree_path])
        file_name += '_'+opath1+".ktx"
        opath2 = "/".join([str(n) for n in block.octree_path])
        output_dir = output_root_folder + "/" + opath2 + "/"
        full_file = output_dir + file_name
        print (full_file)
        if os.path.exists(full_file) and not overwrite:
            file_size = os.stat(full_file).st_size
            if file_size > 0:
                print("Skipping existing file")
                continue
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        f = open(full_file, 'wb')
        try:
            block.write_ktx_file(f)
            f.flush()
            f.close()
        except:
            print("Error writing file %s" % full_file)
            f.close()
            os.unlink(f.name)
            

def convert_subtree_simple(input_path, output_path, subtree_path, subtree_level_count):
    t0 = time.time()
    print( "converting subtree to ktx" )
    specimen = os.path.split(input_path)[1]
    downsample_intensity = True
    downsample_xy = True
    convert_subtree(
            input_root_folder=input_path,
            output_root_folder=output_path + '/' + specimen,
            subtree_tip=subtree_path,
            subtree_level_count=subtree_level_count,
            downsample_intensity=downsample_intensity, 
            downsample_xy=downsample_xy )
    t1 = time.time()
    print ("converting octree subtree to ktx format took %.3f seconds" % (t1 - t0))

if __name__ == "__main__":
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    # octree_path = [5,4,8,3,2,4]
    # octree_path = [6,2,7,3,1,8]
    # octree_path = [1,]
    if sys.argv[3] == '':
        subtree_path = []
    else:
        subtree_path = [int(x) for x in sys.argv[3].split('/')]
    level_count = int(sys.argv[4])
    specimen = os.path.split(input_path)[1]
    print(specimen)
    #
    print(input_path)
    print(output_path)
    print(subtree_path)
    print(level_count)
    convert_subtree_simple(
        input_path=input_path,
        output_path=output_path,
        subtree_path=subtree_path,
        subtree_level_count=level_count,)
