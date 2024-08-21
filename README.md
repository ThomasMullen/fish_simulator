# Fish Tail Simulator

![Pylint](https://github.com/ThomasMullen/zf_animator_tsm/actions/workflows/pylint.yml/badge.svg)
![PythonPackage](https://github.com/ThomasMullen/zf_animator_tsm/actions/workflows/python-package.yml/badge.svg)
![PythonPublish](https://github.com/ThomasMullen/zf_animator_tsm/actions/workflows/python-publish.yml/badge.svg)
![PyPI version](https://badge.fury.io/py/fish-simulator.svg)
[![DOI](https://zenodo.org/badge/683316671.svg)](https://zenodo.org/doi/10.5281/zenodo.11406458)

A Python toolbox for transforming fish tail tracking data into tail animations.

## Description

`fish_simulator` is a Python package used to visualize swimming data from high-speed tracking. 
The typical data used is the tail angle along segments of the tail.
The package converts these tail angles to x-y coordinates and interpolates additional points along the range of tracked tail segments.
These sets of coordinates then transform slices of the fish image, producing a stack of `.png` files that can be rendered into an `mp4` file.

Animations of fish postures are created using a piecewise affine transformation with the interpolated x-y coordinates.

## Package installation

This package can be installed via [PyPi](https://pypi.org/project/fish-simulator/):

```bash
pip install fish-simulator
```

Alternatively, you can download and locally install the package:

```bash
cd path/to/save/package
git clone git@github.com:ThomasMullen/fish_simulator.git
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
On Ubuntu, install ffmpeg via terminal:

```bash
sudo apt install ffmpeg
```

On macOS, install ffmpeg with Homebrew:

```bash
brew install ffmpeg
```

> **NOTE**: Check if `ffmpeg` is installed by running `which ffmpeg` in the terminal. More details can be found [here](https://github.com/kkroening/ffmpeg-python/tree/master).

> **NOTE**: Package can run without `ffmpeg`, but you will not be able to make videos from `.png` files. To avoid dependency issues, set the `vid_fname` argument to `None`.


## Fish animation template

<p align="center">
  <img src="https://github.com/ThomasMullen/fish_simulator/blob/main/fish_imgs/fish_animate.png?raw=true" alt="drawing" width="400"/>
</p>

<p align="center">
  <img src="https://github.com/ThomasMullen/fish_simulator/blob/main/fish_imgs/example_anim.gif?raw=true" alt="drawing" width="400"/>
</p>


## Fish image template

Fish template is an image of the larval zebrafish.

<p align="center">
  <img src="https://github.com/ThomasMullen/fish_simulator/blob/main/fish_imgs/fish.png?raw=true" alt="drawing" width="400"/>
</p>

<p align="center">
  <img src="https://github.com/ThomasMullen/fish_simulator/blob/main/fish_imgs/example_img.gif?raw=true" alt="drawing" width="400"/>
</p>


## Examples

### Simple illustrative plot of posture

This displays a virtual structure of the fish posture. There are several plots you can perform, with the option of converting to a video.
These functions are `plot_bout_elapse`, `plot_skeletal_postures`, and `plot_skeletal_postures_with_trace`. There are passed through the `run` function. Here is an example.

```python
import numpy as np
from fish_simulator.simulator import run, plot_skeletal_postures_with_trace

data_arr = np.load("fish_simulator/test/fixtures/swim01.npy")
run(
    data_arr,
    plot_func=plot_skeletal_postures_with_trace,
    png_dir="path/to/dir/plts",
    vid_fp="path/to/video/anim.mp4",
    line_wid=1
    fps=700,
    )
```

<video src="https://github.com/ThomasMullen/fish_simulator/fish_simulator/examples/demo_vid/run_ani_swim01.mp4" width="640" height="480" controls>
Your browser does not support the video tag.
</video>



https://github.com/ThomasMullen/fish_simulator/assets/38111949/4df1ebc9-e5da-44bd-8c51-1f556040c67e



### Larval zebrafish simulation plot

There are two plotting functions using the real larvae: `plot_tail_image` and `plot_tail_image_with_trace`.

```python
import numpy as np
from fish_simulator.simulator import (
    run,
    plot_tail_image_with_trace,
    plot_tail_image,
)

run(
    data_arr,
    # plot_func=plot_tail_image,
    plot_func=plot_tail_image_with_trace,
    png_dir="path/to/dir/plts",
    vid_fp="path/to/video/anim.mp4",
    fps=700,
)
```

Generate a video from collections of `.png` files

```python
from fish_simulator.utils import make_video

make_video(
    png_dir="path/to/png/files/",
    vid_fname="path/to/video/anim.mp4", 
    framerate=70,
    keep_pngs=True
)
```



https://github.com/ThomasMullen/fish_simulator/assets/38111949/854c1420-c777-4a83-951d-6f31bebe175c



## Citation
```latex
@software{Soares_Mullen_Fish_Behaviour_Simulator_2024,
  author = {Soares Mullen, Thomas},
  month = jun,
  title = {{Fish Tail Simulator}},
  version = {0.1.11},
  year = {2024}
}
```

This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement #813457.


