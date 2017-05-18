# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 15:29:49 2017

@author: Logesh Govindarajulu
"""

#!/usr/bin/env python

# Always prefer setuptools over distutils
from setuptools import setup

# setup platform independent path for the packages
from os import path

here = path.abspath(path.dirname(__file__))

if __name__ == "__main__":
    pkg_name = 'anapy'
    ver = '0.1'
    desc = 'Data analysis for Time series mobile phone data'
    requirements = []
                    
    subpkg_names = [
                    'datamanip',
                    'mlops',
                    'sampling',
                    'externals'
                    ]
                    
    subpkg_path = [pkg_name]
    subpkg_key = [pkg_name]
    # Creating package_dir and packages using package key values and path
    for subpkg in subpkg_names:
        subpkg_path.append(path.join(pkg_name,subpkg))
        subpkg_key.append(pkg_name+'.'+subpkg)
        
    pkg_dir = dict(zip(subpkg_key,subpkg_path))
        
    setup(
        name = pkg_name,
        version = ver,
        description = desc,
        url = 'https://github.com/LogeshGovi/DataAnalysisProject',
        author = 'LogeshGovi',
        author_email = 'logesh.govindarajulu@gmail.com',
        install_requires = requirements,
        package_dir = pkg_dir,
        packages = subpkg_key
    )

