"""Metadata describing the configuration of package"""
from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="fish_simulator",
    version="0.0.4",
    # package_dir={"": "src"},
    packages=find_packages(),
    description="Python toolbox to transform zebrafish\
        tracking data to tail animations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ThomasMullen/fish_simulator",
    author="Thomas Soares Mullen",
    author_email="thomasmullen96@gmail.com",
    license="MIT",
    install_requires=[
        "matplotlib",
        "numpy",
        "h5py",
        "scipy",
        "tqdm",
        "wheel",
        "Pillow",
    ],
    extras_require={
        "dev": ["twine>=4.0.2"],
    },
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development",
        "Topic :: Utilities",
    ],
)
