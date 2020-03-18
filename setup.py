import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "pyBLASTtools",
    version = "0.1",
    author = "Gabriele Coppi",
    description = ("A python package for BLAST-TNG data analysis"),
    packages=find_packages(),
    long_description=read('README.md'),
    python_requires='>=3',
)