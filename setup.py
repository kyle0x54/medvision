#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command


NAME = 'medvision'
DESCRIPTION = 'A python library for medical image computer vision.'
URL = 'https://github.com/kyle0x54/medvision'
EMAIL = 'kyle0x54@163.com'
AUTHOR = 'kyle0x54'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = None


REQUIRED = [
    'matplotlib', 'natsort', 'numpy', 'opencv-python',
    'SimpleITK', 'tqdm', 'visdom'
]

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with open(os.path.join(here, 'README.md')) as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    with open(os.path.join(here, NAME, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION


class UploadCommand(Command):
    """ Support setup.py upload. """

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds...')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution...')
        os.system('{0} setup.py sdist bdist_wheel'.format(sys.executable))
        self.status('Uploading the package to PyPI via Twine...')
        os.system('twine upload dist/*')

        self.status('Pushing git tags...')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')

        sys.exit()


setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=('tests',)),
    install_requires=REQUIRED,
    include_package_data=True,
    license='APACHE',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    cmdclass={
        'upload': UploadCommand,
    },
    zip_safe=False
)
