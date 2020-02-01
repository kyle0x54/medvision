.PHONY: help clean test lint linecount install install_depend

help:
	@echo "Please use 'make <target>' where <target> is one of"
	@echo "  clean-build    to remove build files"
	@echo "  clean-pyc      to remove cache files"
	@echo "  clean          to remove build files and cache files"
	@echo "  test           to run unittests and check code coverage"
	@echo "  lint           to run static analysis of source code"
	@echo "  linecount      to count lines of source code"
	@echo "  install        to install the package in editable mode"
	@echo "  install_depend to install all dependency packages (including test dependencies)"

clean-pyc:
	@find . -type f -name "*.py[co]" | xargs rm -fv
	@find . -type d -name "__pycache__" | xargs rm -rfv
	@find . -type d -name ".pytest_cache" | xargs rm -rfv
	@find . -type f -name ".coverage*" | xargs rm -fv

clean-build:
	@rm -rfv build/
	@rm -rfv dist/
	@rm -rfv *.egg-info
	@find . -type f -name "*.c" | xargs rm -fv
	@find . -type f -name "*.so" | xargs rm -fv

clean: clean-pyc clean-build

test:
	py.test --cov=medvision tests

lint:
	flake8 medvision tests

linecount:
	cloc medvision
	cloc tests

install:
	pip install -e .

install_depend:
	pip install -r requirements.txt
	pip install pytest
	pip install pytest-cov codecov
	pip install flake8

	sudo apt install cloc
