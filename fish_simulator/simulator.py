"""Module pull all functionality and generates the images and videos"""
import os
import itertools
from pathlib import Path
import tempfile
from typing import List, Dict
from tqdm import trange
import numpy as np
from numpy.typing import NDArray
from cycler import cycler
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib import colormaps


from fish_simulator.tail_transformation import (
    interpolate_keypoints,
    compute_angles_from_keypoints,
    convert_tail_angle_to_keypoints,
    KeypointStruct,
)
from fish_simulator.image_loading import ImageLoader


def make_image_simulation(
    data: NDArray,
    upsample: int = None,
    f_path: str = None,
    img_seg_path: str = None,
):
    """simulate fish dynamics on tail image

    Args:
        data (np.ndarray): tail trace
        upsample (int): Increase the segment resolution and interpolation by dividing by upsample.
        f_path (str): filepath to save tmp pngs. Default None, will store and delete the temp folder
        img_seg_path (str): filepath to png segments. Default None, will use default fish

    Example:
    >>> tail_traces = np.load("filepath/tail_angle/data.npy")
    >>> make_image_simulation(data=tail_traces, f_path="/Users/png/dump", upsample=4)
    >>> make_video(png_dir="/Users/png/dump", vid_fname=path/to/video.mp4, keep_pngs=True)

    """
    assert data.ndim == 2, "Need to be 2D"
    if f_path is None:
        f_path = tempfile.mkdtemp()
        print(f"Tmp dir: {f_path}")
    else:
        f_path = Path(f_path)
        f_path.mkdir(parents=True, exist_ok=True)

    data = data.T if data.shape[0] < data.shape[1] else data
    tps = data.shape[0]

    # load image data
    img_loader = ImageLoader(img_seg_path)
    # img_loader = ImageLoader("/Users/tom/VSCode/zf_animator_tsm/src/zf_animator_tsm/\
    #                          template_img/segs")
    head = img_loader.load_head()
    segs = img_loader.load_segments()

    # upsample image
    if upsample is None:
        upsample = 1

    # refine segments
    segs = list(
        itertools.chain(*[np.array_split(seg, upsample, axis=1) for seg in segs])
    )

    # convert angs to x-y coords
    tail_x, tail_y = convert_tail_angle_to_keypoints(
        body_xy=np.zeros((tps, 2)),
        body_angle=np.zeros(tps),
        tail_angle=data,
        body_to_tail_mm=0.5,
        tail_to_tail_mm=0.32,
    )
    # smooth signals
    intp_x, intp_y = interpolate_keypoints(
        KeypointStruct(tail_x, tail_y, len(segs) * upsample)
    )

    # convert pos to angle
    intp_angs, body_ang = compute_angles_from_keypoints(
        body_x=intp_x[:, 0],
        body_y=intp_y[:, 0],
        tail_x=intp_x[:, :],
        tail_y=intp_y[:, :],
    )

    # image hyper parameters
    img_dims = {
        "head_y_len": head.shape[0],
        "head_x_len": head.shape[1],
        "head_to_tail_y_offset": 10,
        "img_sf": 1.0,
    }

    for i_tp in trange(body_ang.size):
        cum_x_len = img_dims["head_x_len"]
        # plot the head
        fig, ax_tail = plt.subplots(dpi=400)
        ax_tail.imshow(
            head,
            extent=[
                0,
                img_dims["head_x_len"],
                -img_dims["head_y_len"] // 2,
                img_dims["head_y_len"] // 2,
            ],
            # transform=rot  + ax_tail.transData,
            transform=ax_tail.transData,
            alpha=0.8,
        )
        # plot segments
        for i, seg in enumerate(segs):
            # add a single segment to scale
            seg_y_len, seg_x_len = seg.shape[:2]
            ax_tail.imshow(
                seg,
                extent=[0, seg_x_len, -seg_y_len // 2, seg_y_len // 2],
                # transform=trs + rot + ax_tail.transData,
                transform=transforms.Affine2D().translate(  # Local translation
                    tx=cum_x_len, ty=img_dims["head_to_tail_y_offset"]
                )
                + transforms.Affine2D().rotate(
                    np.mod(intp_angs[i_tp, i], 2 * np.pi)
                )  # Local rotation
                + ax_tail.transData,  # Global transform
                alpha=0.8,
            )
            cum_x_len += seg_x_len
        ax_tail.spines[["left", "right", "top", "bottom"]].set_visible(False)
        ax_tail.set(
            yticks=[],
            xticks=[],
            xlim=[0, cum_x_len + 20],
            ylim=[(-cum_x_len) * img_dims["img_sf"], cum_x_len * img_dims["img_sf"]],
        )

        fig.savefig(f"{f_path}/{i_tp:03}.png", dpi=150)
        plt.close(fig)

        # Contain in a single function - v slow
        # plot_image_and_segments(
        #     interpolated_angles=intp_angs,
        #     head_img=head,
        #     seg_imgs=segs,
        #     img_dims=img_dims,
        #     f_path=f_path,
        #     i_cntr=i_tp
        #     )


def plot_image_and_segments(
    interpolated_angles: NDArray,
    head_img: NDArray,
    seg_imgs: List[NDArray],
    img_dims: Dict[int, float],
    i_cntr: int,
) -> None:
    """Plot the head an tail warped by interpolated angles

    Args:
        interpolated_angles (NDArray): Interpolated tail angle to transform the img
        head_img (NDArray): image of fish head
        seg_imgs (List[NDArray]): list of image segmentes
        img_dims (Dict[Union[int,float]]): image parameters of dimensions and scale factors
        i_cntr (int): iterator refering to timepoint of image posture.
    """
    fig, ax_tail = plt.subplots(dpi=400)
    # plot the head
    ax_tail.imshow(
        head_img,
        extent=[
            0,
            img_dims["head_x_len"],
            -img_dims["head_y_len"] // 2,
            img_dims["head_y_len"] // 2,
        ],
        # transform=rot  + ax_tail.transData,
        transform=ax_tail.transData,
        alpha=0.8,
    )
    # plot segments
    cum_x_shift = img_dims["head_x_len"]
    for i, seg in enumerate(seg_imgs):
        # add a single segment to scale
        seg_y_len, seg_x_len = seg.shape[:2]
        ax_tail.imshow(
            seg,
            extent=[0, seg_x_len, -seg_y_len // 2, seg_y_len // 2],
            # transform=trs + rot + ax_tail.transData,
            transform=transforms.Affine2D().translate(  # Local translation
                tx=cum_x_shift, ty=img_dims["head_to_tail_y_offset"]
            )
            + transforms.Affine2D().rotate(
                np.mod(interpolated_angles[i_cntr, i], 2 * np.pi)
            )  # Local rotation
            + ax_tail.transData,  # Global transform
            alpha=0.8,
        )
        cum_x_shift += seg_x_len
        # image format
        ax_tail.spines[["left", "right", "top", "bottom"]].set_visible(False)
        ax_tail.set(
            yticks=[],
            xticks=[],
            xlim=[0, cum_x_shift + 20],
            ylim=[
                (-cum_x_shift) * img_dims["img_sf"],
                cum_x_shift * img_dims["img_sf"],
            ],
        )
        plt.close(fig)


def make_posture_simulation(
    data: NDArray,
    png_dir: str,
    n_segments: int = 30,
    line_wid: float = 2.5,
    dpi: int = 150,
):
    """Generate video of tail posture

    Args:
        data (NDArray): 2D timeseries data of angular segments
        png_dir (str): filepath to store individual png plots
        n_segments (int, optional): Number of segments to generate bout. Defaults to 30.
        line_wid (float, optional): Tail posture linewidth. Defaults to 2.5.
        dpi (int, optional): Image resolution. Defaults to 150.
    """
    assert data.ndim == 2, "Need to be 2D"
    if png_dir is None:
        png_dir = tempfile.mkdtemp()
        print(f"Tmp dir: {png_dir}")
    else:
        png_dir = Path(png_dir)
        png_dir.mkdir(parents=True, exist_ok=True)

    data = data.T if data.shape[0] < data.shape[1] else data
    tps = data.shape[0]

    data = data.T if data.shape[0] < data.shape[1] else data
    tps = data.shape[0]
    # convert angs to x-y coords
    tail_x, tail_y = convert_tail_angle_to_keypoints(
        body_xy=np.zeros((tps, 2)),
        body_angle=np.zeros(tps),
        tail_angle=data,
        body_to_tail_mm=0.5,
        tail_to_tail_mm=0.32,
    )
    # smooth signals
    intp_x, intp_y = interpolate_keypoints(KeypointStruct(tail_x, tail_y, n_segments))

    # set frame boundary
    threshold = np.max(np.abs(intp_y))
    for i in trange(tps):
        fig, ax_posture = plt.subplots(figsize=(3, 2))
        ax_posture.spines[["left", "right", "top", "bottom"]].set_visible(False)
        ax_posture.set(yticks=[], xticks=[], ylim=[-threshold, threshold])
        # flip x-axis =  head (left) to tail (right)
        ax_posture.plot(intp_x[i, ::-1], intp_y[i, :], c="k", lw=line_wid)
        fig.savefig(f"{png_dir}/{i:03}.png", dpi=dpi)
        plt.close(fig)


def plot_bout_elapse(
    intp_x: NDArray,
    intp_y: NDArray,
    file_path: str,
    line_wid: float = 1,
    return_color_key: bool = False,
):
    """Plot 2d image of posture elapsed through time and color coded by
    grayscale color map.
    Args:
        intp_x (NDArray): interpolated x values
        intp_y (NDArray): interpolated y values
        file_path (str): filepath to save image
        line_wid (float, optional): linewidth o plot. Defaults to 1.
        return_color_key (bool, optional): _description_. Defaults to False.
    """

    # file_path
    file_path = Path(file_path)
    # define number of timepoints
    tps = intp_x.shape[0]
    # make color cycle through time
    colors = colormaps['gray'](np.linspace(0.1, 0.99, tps))

    fig, ax_tail = plt.subplots(figsize=(3, 2), dpi=150)
    ax_tail.set_prop_cycle(cycler(color=colors))
    for i in trange(tps):
        # flip x-axis - head (left) to tail (right)
        ax_tail.plot(intp_x[i, ::-1], intp_y[i, :], lw=line_wid, alpha=0.8)
    ax_tail.spines[["left", "right", "top", "bottom"]].set_visible(False)
    ax_tail.set(
        yticks=[], xticks=[], ylim=[-np.max(np.abs(intp_y)), np.max(np.abs(intp_y))]
    )
    fig.savefig(file_path, transparent=True, dpi=350, bbox_inches="tight")

    # return the colro encoding time
    if return_color_key:
        col_key = np.linspace(np.min(intp_x), np.max(intp_x), tps)
        fig, ax_tail = plt.subplots(figsize=(3, 0.3), dpi=150)
        for i in range(tps - 1):
            ax_tail.spines[["left", "right", "top", "bottom"]].set_visible(False)
            ax_tail.set(
                yticks=[],
                xticks=[],
            )
            ax_tail.hlines(
                0, col_key[i], col_key[i + 1], lw=2, alpha=1, color=colors[i]
            )
        fig.savefig(
            Path(file_path.parent, f"{file_path.stem}_key{file_path.suffix}"),
            transparent=True,
            dpi=350,
            bbox_inches="tight",
        )


def make_video(png_dir: str, vid_fname: str, keep_pngs: bool = True) -> None:
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
    cmd = f"ffmpeg -r 35 -f image2 -i '{png_dir}'/%03d.png -vcodec libx264 \
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
