#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from setuptools import find_packages, setup

NAME = "medvision"
DESCRIPTION = "A python library for medical image computer vision."
URL = "https://github.com/kyle0x54/medvision"
EMAIL = "kyle0x54@163.com"
AUTHOR = "kyle0x54"
REQUIRES_PYTHON = ">=3.6.0"
VERSION = None


REQUIRED = [
    "matplotlib",
    "natsort",
    "numpy",
    "opencv-python",
    "pandas",
    "pillow",
    "pycryptodome",
    "pydicom",
    "scikit-learn",
    "SimpleITK",
    "tqdm",
]

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with open(os.path.join(here, "README.md")) as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    with open(os.path.join(here, NAME, "__version__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION

setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=("tests",)),
    install_requires=REQUIRED,
    include_package_data=True,
    license="APACHE",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    zip_safe=False,
)
