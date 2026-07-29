"""Compatibility entry point for older editable-install tooling.

Package metadata, dependencies, resources, and console scripts live in
``pyproject.toml``. Modern installers do not need to execute this file
directly.
"""

from setuptools import setup


setup()
