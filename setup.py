#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

def calculate_version():
    initpy = open('LocallyWeightedFusion/_version.py').read().split('\n')
    version = list(filter(lambda x: '__version__' in x, initpy))[0].split('\'')[1]
    return version

package_version = calculate_version()

setup(
    name='LocallyWeightedFusion',
    version=package_version,
    author='William La Cava',
    author_email='lacava@upenn.edu',
    packages=find_packages(),
    url='https://github.com/lacava/LocallyWeightedFusion',
    download_url='https://github.com/lacava/LocallyWeightedFusion/releases/tag/'+package_version,
    license='GNU/GPLv3',
    entry_points={'console_scripts': ['LocallyWeightedFusion=LocallyWeightedFusion:main', ]},
    # test_suite='nose.collector',
    # tests_require=['nose'],
    description=('Locally Weighted Fusion'),
    long_description='''
A simple distance-based classification algorithm.

Contact:
===
e-mail: lacava@upenn.edu

This project is hosted at https://github.com/lacava/LocallyWeightedFusion
''',
    zip_safe=True,
    install_requires=['numpy', 'scipy', 'pandas', 'scikit-learn'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        # 'Programming Language :: Python :: 2.7',
        # 'Programming Language :: Python :: 3',
        # 'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    keywords=['data science', 'machine learning', 'ensemble learning', 'classification', 'regression'],
)
