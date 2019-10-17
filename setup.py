# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
    name="rdb",
    version="0.0.1",
    author="Zhiyang He",
    author_email="hzyjerry@berkeley.edu",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "pyglet==1.3.2",
        "numpy",
        "matplotlib",
        "jax",
        "jaxlib",
        "gym",
        "moviepy",
        "pygame",
        "toolz",
        "PyOpenGL",
    ],
)
