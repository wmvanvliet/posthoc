#! /usr/bin/env python
from setuptools import setup
import os

if __name__ == "__main__":
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    setup(name='posthoc',
          maintainer='Marijn van Vliet',
          maintainer_email='w.m.vanvliet@gmail.com',
          description='Post-hoc modification of linear models',
          license='BSD-3',
          url='https://github.com/wmvanvliet/posthoc',
          version='0.1',
          download_url='https://github.com/wmvanvliet/posthoc/archive/master.zip',
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
          packages=['posthoc'],
          install_requires=['numpy', 'scipy', 'scikit-learn', 'progressbar2'])
