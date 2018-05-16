#! /usr/bin/env python
from setuptools import setup
import os

if __name__ == "__main__":
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    setup(name='workbench',
          maintainer='Marijn van Vliet',
          maintainer_email='w.m.vanvliet@gmail.com',
          description='A workbench for linear models',
          license='BSD-3',
          url='https://version.aalto.fi/gitlab/vanvlm1/workbench',
          version='0.1',
          download_url='https://version.aalto.fi/gitlab/vanvlm1/workbench/repository/archive.zip?ref=master',
          long_description=open('README.md').read(),
          classifiers=['Intended Audience :: Science/Research',
                       'Intended Audience :: Developers',
                       'License :: OSI Approved',
                       'Programming Language :: Python',
                       'Topic :: Software Development',
                       'Topic :: Scientific/Engineering',
                       'Operating System :: Microsoft :: Windows',
                       'Operating System :: POSIX',
                       'Operating System :: Unix',
                       'Operating System :: MacOS'],
          platforms='any',
          packages=['workbench'],
          install_requires=['numpy', 'scipy', 'scikit-learn'])
