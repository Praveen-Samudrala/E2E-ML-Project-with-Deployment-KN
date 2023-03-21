''' To create the application as a Python Package'''

from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(path:str) -> List:
    '''
    this function takes in the requirements.txt file path
    and returns the list pf requirements'''
    requirements = []
    with open(path, 'r') as f:
        requirements = f.readlines()
        requirements = [req.replace("\n"," ") for req in requirements]

        if HYPHEN_E_DOT in requirements: requirements.remove(HYPHEN_E_DOT)

setup(
    name='mlprojectPraveen',
    version='0.1',
    author='Praveen',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')

)