'''
Created on Sep 29, 2016

@author: brunsc
'''

import os
import time

from ktx.octree.ktx_from_rendered_tiff import RenderedMouseLightOctree, RenderedTiffBlock

def convert_one_octree_block(
        root_folder, 
        octree_path=[], 
        downsample_intensity=True, 
        downsample_xy=True, 
        file_name="converted.ktx"):
    o = RenderedMouseLightOctree(
            # input_folder=os.path.abspath('./practice_octree_input'), 
            input_folder=os.path.abspath(root_folder), 
            downsample_intensity=downsample_intensity,
            downsample_xy=downsample_xy)
    subfolder = os.path.sep.join([str(n) for n in octree_path])
    folder = os.path.join(root_folder, subfolder)
    b = RenderedTiffBlock(folder, o, octree_path)
    f = open(file_name, 'wb')
    b.write_ktx_file(f)
    f.flush()
    f.close()
    do_lz4_compress = False
    if do_lz4_compress:
        cmd = "LZ4.exe %s > %s.lz4" % (file_name, file_name)
        print (cmd)
        os.system(cmd)
        os.remove(file_name) # Delete uncompressed version
    
if __name__ == "__main__":
    t0 = time.time()
    print( "converting file to ktx" )
    specimen = '2016-04-04b'
    downsample_intensity = True
    downsample_xy = True
    octree_path = [5,4,8,3,2,4]
    # octree_path = [6,2,7,3,1,8]
    # octree_path = [1,]
    file_name = 'block_' + specimen
    if downsample_intensity:
        file_name += '_8'
    if downsample_xy:
        file_name += '_xy'
    file_name += '_'+''.join([str(n) for n in octree_path])+".ktx"
    convert_one_octree_block(
            root_folder='/nobackup2/mouselight/' + specimen,
            octree_path=octree_path,
            file_name=file_name,
            downsample_intensity=downsample_intensity, 
            downsample_xy=downsample_xy )
    t1 = time.time()
    print ("converting octree tiff file to ktx format took %.3f seconds" % (t1 - t0))
