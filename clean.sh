#! /bin/bash

rm -rf build/  dist/ *.egg-info
find . -type d -name '__pycache__' -exec rm -rf {} \;
find . -type f -name '*.pyc' -delete
