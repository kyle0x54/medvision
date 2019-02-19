#! /bin/bash

rm -rf build/  dist/ *.egg-info
find . -type f -name ".coverage*" -delete
find . -type f -name "*.py[co]" -delete
find . -type d -name "__pycache__" -delete
