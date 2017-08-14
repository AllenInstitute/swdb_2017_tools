#!/usr/bin/env python
# -*- coding: utf-8 -*-
import setuptools

"""The setup script."""

from setuptools import setup, find_packages

description="A collaborative Python package built by participants of the Summer Workshop on the Dynamic Brain",
long_description=open('README.md').read(),
with open('README.md') as readme_file:
    readme = readme_file.read()

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

with open('test_requirements.txt','r') as f:
    test_requirements = f.read().splitlines()

setup(
    name='swdb_2017_tools',
    version='0.1.0',
    description="Summer Workshop on the Dynamic Brain 2017 Tools.",
    long_description=readme,
    author="Summer Workshop for Dynamic Brain 2017 Participants",
    author_email='nicholasc@alleninstitute.org',
    url='https://github.com/nicain/swdb_2017_tools',
    packages=find_packages(include=['swdb17']),
    include_package_data=True,
    install_requires=requirements,
    license="BSD license",
    zip_safe=False,
    keywords='swdb_2017_tools',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.7',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
