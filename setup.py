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
    subpkg_names = ['datamanip', 'mlops','sampling']
    subpkg_path = [pkg_name]
    subpkg_key = [pkg_name]
    for subpkg in subpkg_names:
        subpkg_path.append(path.join(pkg_name,subpkg))
        subpkg_key.append(pkg_name+'.'+subpkg)
        
    pkg_dir = dict(zip(subpkg_key,subpkg_path))
        
    setup(
        name = pkg_name,
        package_dir = pkg_dir,
        packages = subpkg_key
    )

