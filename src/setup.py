#!/usr/bin/env python

from setuptools import find_packages, setup


def get_version(rel_path):
    """Extract version information from source code."""
    with open(rel_path, "r") as file:
        for line in file:
            if line.startswith('__version__'):
                # Remove quotation marks
                delims = "\"'"
                return line.split('=')[1].strip().strip(delims)
    return "0.0.0"  # Default if version not found


# TODO @GiacomoFabrini have no clue what to put here!
setup(
   name='CurriculumMultipleShooting_NeurIPS_2026',
   version=get_version("__init__.py"),
   description='Benchmarking diffrax and PEtab.jl on UDE models',
   author='Giacomo Fabrini, Sebastian Persson',
   author_email='giacomo.fabrini@crick.ac.uk, sebpe@chalmers.se',
   python_requires=">=3.12",
   packages=find_packages(),
   include_package_data=True,
)
