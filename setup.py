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
        "Pillow==7.2.0",
        "scipy",
        "scikit-image",
        # "scipy==1.1.0",
        "pyglet==1.5.11",
        "matplotlib",
        "gym",
        "flax",
        "numpyro==0.10.1",
        "jax==0.4.1",
        "jaxlib==0.4.1",
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
        "imageio==2.5",
        "ipywidgets",
        "flask",
        "imageio-ffmpeg",
        "tensorboardX",
    ],
)
