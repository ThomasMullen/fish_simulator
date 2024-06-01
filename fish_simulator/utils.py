"""Helper functions for sorting data/files"""
import os
from pathlib import Path
import tempfile
from typing import Tuple, Union
import numpy as np
from numpy.typing import NDArray
import matplotlib.colors as mc
import colorsys
from cycler import cycler
import cmcrameri.cm as cmc


grey_to_black_cycler = cycler(color=cmc.grayC(np.linspace(0.9, 0.1, 10)))


def orientate_data(data: NDArray) -> Tuple[NDArray, Tuple[int, int]]:
    """reorientate data s.t. axis 0 is time and axis 1 is segments.

    Args:
        data (NDArray): 2D timeseries

    Returns:
        Tuple[NDArray, Tuple[int, int]]: rotated data and tuple (tps, n)
    """
    assert data.ndim == 2, "Need to be 2D data"
    data = data.T if data.shape[0] < data.shape[1] else data
    return data, data.shape


def make_dir(fp: Union[Path, str]) -> Path:
    """Create a directory at the given file path.

    Args:
        fp (Union[Path, str]): The file path where the directory should be created.

    Returns:
        Path: The path of the created directory.
    """
    if fp is None:
        fp = tempfile.mkdtemp()
        print(f"Tmp dir: {fp}")
    else:
        fp = Path(fp)
        fp.mkdir(parents=True, exist_ok=True)
    return Path(fp)


def make_video(
    png_dir: str, vid_fname: str, framerate: int = 35, keep_pngs: bool = True
) -> None:
    """converts the save png figs to mp4 with ffmpeg

    Args:
        png_dir (str): directory with numbered pngs
        vid_fname (str): video filepath
        keep_pngs (bool, optional): delete dir with png after video saved.
        Defaults to True.
    """
    vid_fname = Path(vid_fname)
    png_dir = Path(png_dir)
    if vid_fname.exists():
        os.remove(vid_fname)

    # make video
    cmd = f"ffmpeg -r {framerate} -f image2 -i '{png_dir}'/%05d.png -vcodec libx264 \
    -crf 25 -pix_fmt yuv420p '{vid_fname}'"
    os.system(cmd)
    print(f"Saving video to: {vid_fname}")

    if not keep_pngs:
        try:
            os.remove(png_dir)
            print(f"delete: {png_dir} folder")
        # except OSError as e:
        #     raise PermissionError(f"Not permitted to delete dir:\n{png_dir}") from e
        except PermissionError as perm_e:
            print(f"Not permitted to delete dir:\n{png_dir}\n{perm_e}")


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def make_color_cycle(color: NDArray, n_colors: int, reverse: bool = False):
    """Make a matplotlib color cycle

    Args:
        color (np.array): color want to split into light-dark
        n_colors (int): number of different colors
        reverse (bool, optional): reverse the color cycle. Defaults to False.

    Returns:
        dict: color wheel

    Examples:
    >> c_cycle = make_color_cycle(color=np.array(), n_colors=7, reverse=False)
    >> fig, ax = plt.subplots()
    >> ax.set_prop_cycle(c_cycle)

    """
    if reverse:
        colour_cycler = cycler(
            color=[lighten_color(color, i) for i in np.linspace(0.9, 0.2, n_colors)]
        )
    else:
        colour_cycler = cycler(
            color=[lighten_color(color, i) for i in np.linspace(0.2, 0.9, n_colors)]
        )
    return colour_cycler
