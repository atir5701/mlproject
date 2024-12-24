# The setup.py file is an essential component of Python projects intended for distribution.
#  It provides metadata and configuration for a Python package and defines how it should be built, installed, and managed.
#  It's typically used with the setuptools library and is essential for publishing packages to the Python Package Index (PyPI).


# By providing setup.py we can allow our machine learning application to be used as a package and can be added to the PyPI.

from setuptools import setup, find_packages
from typing import List

def get_packages(file_path):
    '''
    This function will return the list of packages needed for the project.
    '''
    req=[]
    with open(file_path) as file_obj:
        req = file_obj.readlines()
        req=[i.replace("\n","")  for i in req]

        if '-e .' in req:
            req.remove('-e .')

    return req



setup(
    name="mlproject",
    version='1.0.0',
    author="Atir Shakhrelia",
    author_email='atirsakhrelia@gmail.com',
    packages=find_packages(),
    install_required=get_packages('requirements.txt')
)