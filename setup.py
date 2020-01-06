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
        "pillow",
        "scipy==1.1.0",
        "pyglet==1.3.2",
        "numpyro",
        "matplotlib",
        "jax",
        "jaxlib",
        "gym",
        "moviepy",
        "pygame",
        "toolz",
        "funcy",
        "PyOpenGL",
        "seaborn",
        "ray",
        "psutil",
        "tqdm",
    ],
)
