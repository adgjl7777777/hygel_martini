
from setuptools import setup, find_packages

setup(
    name="hygel_martini",
    version="0.1.0",
    author="Gemini",
    description="A package for generating hydrogel structures for molecular dynamics simulations.",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    install_requires=[
    'numpy',
    'scipy',
    'numba',
    ],
)
