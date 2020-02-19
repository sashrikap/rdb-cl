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
        "matplotlib",
        "gym",
        "numpyro @ https://github.com/hzyjerry/numpyro/tarball/master#egg=numpyro"
        "jax==0.1.54",
        "jaxlib==0.1.38",  # numpyro installs the correct version
        "moviepy",
        "fast-histogram",
        "pygame",
        "toolz",
        "funcy",
        "PyOpenGL",
        "seaborn",
        "ray",
        "ipython",
        "tqdm",
        "imageio",
        "ipywidgets",
    ],
)
