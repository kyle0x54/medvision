[![Build Status](https://img.shields.io/travis/kyle0x54/medvision.svg?label=Linux%20build%20%40%20Travis%20CI&style=flat)](https://travis-ci.org/kyle0x54/medvision)
[![PyPI](https://img.shields.io/pypi/v/medvision.svg?colorB=blue&style=flat)](https://pypi.org/project/medvision/)
[![PyVersions](https://img.shields.io/pypi/pyversions/medvision.svg?style=flat)](https://pypi.org/project/medvision/)
[![codecov](https://codecov.io/gh/kyle0x54/medvision/branch/master/graph/badge.svg)](https://codecov.io/gh/kyle0x54/medvision)
[![Maintainability](https://api.codeclimate.com/v1/badges/8907b3e1989a12585139/maintainability)](https://codeclimate.com/github/kyle0x54/medvision/maintainability)
[![DOI](https://zenodo.org/badge/167765585.svg)](https://zenodo.org/badge/latestdoi/167765585)
[![GitHub license](https://img.shields.io/github/license/kyle0x54/medvision.svg?style=flat)](https://github.com/kyle0x54/medvision/blob/master/LICENSE)

**medvision** is an open source python library for medical computer vision.

## Installation

*Install dependencies*

```shell
$ conda install -c conda-forge gdcm
```

*Install medvision*

```shell
$ git clone git@github.com:kyle0x54/medvision.git
$ cd medvision
$ pip install -r requirements.txt
$ pip install -e .
```

## Try your first medvision program

```shell
$ python
```

```python
>>> import medvision as mv
>>> mv.isdicom('001.dcm')
True
```

## License

[Apache License 2.0](LICENSE)
