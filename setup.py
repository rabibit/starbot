#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import os

from setuptools import setup

NAME = 'starbot'
DESCRIPTION = 'A robot for IVR'
URL = 'https://github.com/kvinlwang/foolbot'
EMAIL = 'wy721@qq.com'
AUTHOR = 'Kevin Wang'
VERSION = '0.1.0'

REQUIRED = ["rasa", "tensorflow"]

EXTRAS = {}


here = os.path.abspath(os.path.dirname(__file__))

try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION


setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=['starbot'],
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
)
