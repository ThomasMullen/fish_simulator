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
pip install -e .
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

![fish_ani_template](https://github.com/ThomasMullen/fish_simulator/fish_imgs/fish_animate.png)


## Default Fish image

![fish_template](https://github.com/ThomasMullen/fish_simulator/fish_imgs/fish.png)


![zebrafish_example](https://github.com/ThomasMullen/fish_simulator/fish_imgs/example_sim.gif)

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

### Real larval zebrafish simulation plot

There are two plotting functions using the real larvae: `plot_tail_image` and `plot_tail_image_with_trace`.

```python
import numpy as np
from fish_simulator import run, plot_tail_image_with_trace, plot_tail_image

run(
    data_arr,
    plot_func=plot_tail_image_with_trace,
    # plot_func=plot_tail_image,
    dir="path/to/dir/plts",
    vid_fp="path/to/video/anim.mp4",
    n_intp_segs=40,
    img_kwargs={"body_to_tail_mm": 156.3, "tail_to_tail_mm": -181.3},
    line_wid=1,
)
```

<video src="https://github.com/ThomasMullen/fish_simulator/examples/demo_vid/run_img_swim01.mp4" width="640" height="480" controls>
Your browser does not support the video tag.
</video>

### Generate a simulation of the fish tail posture

```python
from fish_simulator.simulator import make_posture_simulation, make_video
make_posture_simulation(data=tail_angle_data, n_segments=30, png_dir="dir/to/save/png_files")
make_video(png_dir="dir/to/save/png_files", vid_fname="file/path/of/generated_vid.mp4")
```

## Key Functions
