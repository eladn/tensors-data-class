#!/usr/bin/env

import setuptools

"""
>>> python setup.py bdist_wheel
>>> python -m pip install dist/tensors_data_class-<XX.XX>-py3-none-any.whl
>>> python -m twine upload dist/*
"""

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='tensors_data_class',
    version='0.0.1',
    provides=['tensors_data_class'],
    py_modules=['tensors_data_class'],
    author="Elad Nachmias",
    author_email="eladnah@gmail.com",
    description="PyTorch Extension Library for organizing tensors in a form "
                "of a structured tree of dataclasses, with built-in support "
                "for advanced collating mechanisms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eladn/tensors-data-class",
    packages=setuptools.find_packages(),
    install_requires=['pytorch'],
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
 )
