#!/bin/env python

'''
Created on Aug 9, 2016

@author: Christopher Bruns
'''

from setuptools import setup

# Load module version from ovr/version.py
exec(open('src/ktx/version.py').read())

setup(
    name = "pyktx",
    version = __version__,
    author = "Christopher Bruns",
    author_email = "cmbruns@rotatingpenguin.com",
    description = "Pure python tools for reading and writing KTX format OpenGL image texture files",
    url = "https://github.com/cmbruns/pyktx",
    download_url = "https://github.com/cmbruns/pyktx/tarball/" + __version__,
    package_dir = {'': 'src'},
    packages = ['ktx',],
    keywords = "opengl ktx texture file format",
    classifiers = [],
)
