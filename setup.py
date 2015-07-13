#!/usr/bin/env python
import sys
import os

if "publish" in sys.argv[-1]:
    os.system("python setup.py sdist upload")
    sys.exit()

try:
    from setuptools import setup
    setup
except ImportError:
    from distutils.core import setup
    setup

setup(name='pysco',
      version='0.1',
      description='Analysis code for kernel phase and non-redundant masking',
      author='Benjamin Pope',
      author_email='benjamin.pope@astro.ox.ac.uk',
      url='https://github.com/benjaminpope/pysco',
      packages=['pysco'],
      install_requires=['numpy','matplotlib','pyfits'])
