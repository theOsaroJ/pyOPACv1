# setup.py

from setuptools import setup, find_packages

setup(
    name='opac3',
    version='1.0.0',
    description='Molecular property prediction',
    author='Etinosa Osaro',
    affiliation='Colon Lab @ University of Notre Dame',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'torch',
        'torchvision',
        'torch_geometric',
        'rdkit-pypi',
        'ase',
        'matplotlib',
        'tqdm',
    ],
)
