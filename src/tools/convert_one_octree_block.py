'''
Created on Sep 29, 2016

@author: brunsc
'''

import os

import libtiff

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
    cmd = "LZ4.exe %s > %s.lz4" % (file_name, file_name)
    print (cmd)
    os.system(cmd)
    os.remove(file_name) # Delete uncompressed version
    
if __name__ == "__main__":
    libtiff.libtiff_ctypes.suppress_warnings()
    print( "converting file to ktx" )
    octree_path = [5,4,8,3,2,4]
    for downsample_intensity in (True,False):
        for downsample_xy in (True,False):
            file_name = 'block20160404b_'
            if downsample_intensity:
                file_name += '8'
            if downsample_xy:
                file_name += 'xy'
            file_name += '_'+''.join([str(n) for n in octree_path])+".ktx"
            convert_one_octree_block(
                    root_folder='//fxt/nobackup2/mouselight/2016-04-04b',
                    octree_path=octree_path,
                    file_name=file_name,
                    downsample_intensity=downsample_intensity, 
                    downsample_xy=downsample_xy )
