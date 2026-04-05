
from setuptools import setup, find_packages

setup(
    name="hygel_martini",
    version="0.1.0",
    author="Daehong Kim",
    description="A package for generating hydrogel structures for molecular dynamics simulations.",
    packages=find_packages(),
    package_data={
        "hygel_martini.param_opt.qm_to_martini": ["xtb_traj_to_pdb.py"],
        "hygel_martini": ["bash_settings/launcher_utils.sh"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    install_requires=[
    'numpy',
    'scipy',
    'numba',
    'ase',
    'CoolProp',
    'PyYAML',
    'requests',
    ],
)
