#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='research_seed',
      version='0.0.1',
      description='Rethinking-Binarized-Neural-Network-Optimization Replication',
      author='',
      author_email='',
      url='https://github.com/bsridatta/Rethinking-Binarized-Neural-Network-Optimization',  
      install_requires=[
            'pytorch-lightning'
      ],
      packages=find_packages()
      )

