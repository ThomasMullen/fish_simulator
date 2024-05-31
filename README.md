# Zebrafish Tail Animator

![Pylint](https://github.com/ThomasMullen/zf_animator_tsm/actions/workflows/pylint.yml/badge.svg)
![PythonPackage](https://github.com/ThomasMullen/zf_animator_tsm/actions/workflows/python-package.yml/badge.svg)
![PythonPublish](https://github.com/ThomasMullen/zf_animator_tsm/actions/workflows/python-publish.yml/badge.svg)
![PyPI version](https://badge.fury.io/py/fish-simulator.svg)

Python toolbox to transform zebrafish tracking data to tail animations

## Description

`zf_animator` is a Python package used to create visualisation of how swimming data looks from high-speed tracking data.
The typical data used should be the tail angle along segments of the tail.
The package will convert these tail angles to x- y- coordinates, and interpolate many more points along the range of tracked tail segments.
These sets of coordinates will then transform slices of the zebrafish image, producing a stack of .png files that can be rendered into a mp4 file.

Animations of the larval zebrafish postures are warped using a piece wise affine transformation using the interpolated x-y coordinates.


## Package installation

This package can be installed via pip

```bash
pip install fish-simulator
```

Alternatively, you can download and locally install the package.

```bash
cd path/to/save/package
git clone clone-latest-tag git@github.com:ThomasMullen/fish_simulator.git
cd ./fish_simulator
python -m build
# go to working project
cd path/to/work/project
# acivate venv
source my_virtual_env/bin/activate
pip install -e path/to/save/package
```

## Requires `ffmpeg`

### Installation instructions

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

## Default Fish animation

<p align="center">
  <img src="https://github.com/ThomasMullen/fish_simulator/blob/main/fish_imgs/fish_animate.png?raw=true" alt="drawing" width="400"/>
</p>

<p align="center">
  <img src="https://github.com/ThomasMullen/fish_simulator/blob/main/fish_imgs/example_anim.gif?raw=true" alt="drawing" width="400"/>
</p>


## Default Fish image

<p align="center">
  <img src="https://github.com/ThomasMullen/fish_simulator/blob/main/fish_imgs/fish.png?raw=true" alt="drawing" width="400"/>
</p>

<p align="center">
  <img src="https://github.com/ThomasMullen/fish_simulator/blob/main/fish_imgs/example_img.gif?raw=true" alt="drawing" width="400"/>
</p>


## Example

### Simple illustrative plot of posture

This displays a virtual structure of the fish posture. There are several plots you can perform, with the option of converting to a video.
These functions are `plot_bout_elapse`, `plot_skeletal_postures`, and `plot_skeletal_postures_with_trace`. There are passed through the `run` function. Here is an example.

```python
import numpy as np
from fish_simulator import run

data_arr = np.load("fish_simulator/test/fixtures/swim01.npy")
run(
    data_arr,
    plot_func=plot_skeletal_postures_with_trace,
    dir="path/to/dir/plts",
    vid_fp="path/to/video/anim.mp4",
    line_wid=1
    )
```

<video src="https://github.com/ThomasMullen/fish_simulator/fish_simulator/examples/demo_vid/run_ani_swim01.mp4" width="640" height="480" controls>
Your browser does not support the video tag.
</video>



https://github.com/ThomasMullen/fish_simulator/assets/38111949/4df1ebc9-e5da-44bd-8c51-1f556040c67e



### Real larval zebrafish simulation plot

There are two plotting functions using the real larvae: `plot_tail_image` and `plot_tail_image_with_trace`.

```python
import numpy as np
from fish_simulator import run, plot_tail_image_with_trace, plot_tail_image

run(
    data_arr,
    # plot_func=plot_tail_image,
    plot_func=plot_tail_image_with_trace,
    dir="path/to/dir/plts",
    vid_fp="path/to/video/anim.mp4",
)
```

Generate a video from collections of `.png` files

```python
from utils import make_video

make_video(
    png_dir="path/to/png/files/",
    vid_fname="path/to/video/anim.mp4", 
    framerate=70,
    keep_pngs=True
)
```



https://github.com/ThomasMullen/fish_simulator/assets/38111949/854c1420-c777-4a83-951d-6f31bebe175c



## Key Functions
