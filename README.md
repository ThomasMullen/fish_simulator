# Zebrafish Tail Animator
![Pylint](https://github.com/ThomasMullen/zf_animator_tsm/actions/workflows/pylint.yml/badge.svg)
![PythonPackage](https://github.com/ThomasMullen/zf_animator_tsm/actions/workflows/python-package.yml/badge.svg)
![PythonPublish](https://github.com/ThomasMullen/zf_animator_tsm/actions/workflows/python-publish.yml/badge.svg)
![PyPI version](https://badge.fury.io/py/zf_animator.svg)

Python toolbox to transform zebrafish tracking data to tail animations

## Description
`zf_animator` is a Python package used to create visualisation of how swimming data looks from high-speed tracking data.
The typical data used should be the tail angle along segments of the tail.
The package will convert these tail angles to x- y- coordinates, and interpolate many more points along the range of tracked tail segments.
These sets of coordinates will then transform slices of the zebrafish image, producing a stack of .png files that can be rendered into an mp4 file.

## Installation
To convert the `.png` files to an `.mp4` animation requires `ffmpeg` to be installed on the running computer.
This can be install in Ubuntu in the terminal:

```bash
sudo apt install ffmpeg
```

Or with OS X can be installed:
```bash
brew install ffmpeg
```

Check it is installed by running `which ffmpeg` in the terminal. More details can be found [here](https://github.com/kkroening/ffmpeg-python/tree/master).

### Dependencies

## Default zebrafish image
**Ref the source of the image**

## Example

## Key Functions




