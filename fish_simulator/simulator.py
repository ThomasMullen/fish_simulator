"""Module pull all functionality and generates the images and videos"""
import os
import itertools
from pathlib import Path
import tempfile
from typing import List, Dict, Tuple, Callable
from tqdm import trange
from numpy.typing import NDArray
import numpy as np
from scipy.interpolate import interp1d
from cycler import cycler
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib import colormaps

from fish_simulator.transforms import (
    interpolate_keypoints,
    compute_angles_from_keypoints,
    convert_tail_angle_to_keypoints,
    KeypointStruct,
)
from fish_simulator.image_loader import ImageLoader, PostureStruct
from fish_simulator.utils import (
    orientate_data,
    make_dir,
    make_video,
    make_color_cycle,
    grey_to_black_cycler,
)


def run(data: NDArray, plot_func: Callable, dir: str, vid_fp: str, **kwargs):
    intp_xy, (low_xy, upp_xy), _ = generate_skeletal_postures(data)
    plot_func(data, low_xy, intp_xy, upp_xy, dir, **kwargs)
    if vid_fp is not None:
        vid_fp = Path(vid_fp)
        make_video(dir, vid_fp)
    pass


def generate_skeletal_postures(
    data: NDArray,
    posture_struct: PostureStruct = PostureStruct(),
    intp_n_segs: int = 30,
) -> Tuple[NDArray, Tuple[NDArray, NDArray], NDArray]:
    """Generate skeletal postures based on input data.

    This function takes in input data, file path, posture structure, and the number of interpolation segments as parameters.
    It generates skeletal postures by performing various calculations and interpolations on the input data.

    Args:
        data (NDArray): The input data.
        f_path (str): The file path.
        posture_struct (PostureStruct, optional): The posture structure. Defaults to PostureStruct().
        intp_n_segs (int, optional): The number of interpolation segments. Defaults to 30.

    Returns:
        Tuple[NDArray, Tuple[NDArray, NDArray], NDArray]: A tuple containing the interpolated x-y positions, the lower and upper positions, and the normal directions.
    """
    data, (tps, n_dims) = orientate_data(data)

    # convert x-y pos
    tail_x, tail_y = convert_tail_angle_to_keypoints(
        body_xy=np.zeros((tps, 2)),
        body_angle=np.zeros(tps),
        tail_angle=data * -1.2,  # scale factor to account for tail tip
        body_to_tail_mm=0.5,
        tail_to_tail_mm=0.32,
    )

    # interpolate x-y pos
    intp_xy = np.array(
        interpolate_keypoints(KeypointStruct(tail_x, tail_y, intp_n_segs))
    )
    # interpolate width
    interpolator_width = interp1d(
        np.linspace(0, 1, posture_struct.n_segs + 1),
        posture_struct.seg_width,
        kind="cubic",
        axis=0,
    )
    intp_seg_width = interpolator_width(np.linspace(0, 1, intp_n_segs + 1))
    # calculate tangential direction
    tang_xy = np.c_[
        np.c_[-np.ones(tps), np.zeros(tps)].T[..., np.newaxis], np.diff(intp_xy, axis=2)
    ]
    # calculate normal direction
    n_xy = np.flip(tang_xy, axis=0)
    n_xy[0] *= -1
    # normalise
    n_xy_norm = np.linalg.norm(n_xy, axis=0)
    n_xy /= n_xy_norm
    # update upper and lower pos
    upp_xy = intp_xy - (n_xy * intp_seg_width / 2)
    low_xy = intp_xy + (n_xy * intp_seg_width / 2)

    return intp_xy, (low_xy, upp_xy), n_xy


def plot_skeletal_postures_with_trace(
    trace_data: NDArray,
    lower: NDArray,
    center: NDArray,
    upper: NDArray,
    f_path: str,
    posture_struct: PostureStruct = PostureStruct(),
    fps=700,
    line_wid=2,
):
    f_path = make_dir(f_path)
    trace_data, (tps, n_dims) = orientate_data(trace_data)
    # set frame boundary
    threshold = np.max(np.abs(center[1]))
    time_ms = np.arange(tps) * 1000 / fps
    n_segs = lower.shape[-1]

    for t_ in trange(tps):
        fig, (ax_trace, ax_posture) = plt.subplots(1, 2, figsize=(8, 4), dpi=400)
        # fish_trace
        ax_trace.set_title(f_path.stem)
        ax_trace.set_prop_cycle(grey_to_black_cycler)
        ax_trace.plot(time_ms[:t_], trace_data[:t_], alpha=1)
        ax_trace.set_xlim(0, time_ms[-1])
        ax_trace.set_ylim(-2.4, 4.0)
        ax_trace.set_xticks([])
        ax_trace.set_yticks([])
        # ax_trace.spines[['right', 'top']].set_visible(True)
        ax_trace.set_axis_off()

        ax_posture.plot(upper[0, t_, :], upper[1, t_, :], c="k", lw=line_wid)
        ax_posture.plot(center[0, t_, :], center[1, t_, :], c="k", lw=line_wid)
        ax_posture.plot(lower[0, t_, :], lower[1, t_, :], c="k", lw=line_wid)
        for i in range(0, n_segs + 1, (n_segs + 1) // 10):
            ax_posture.plot(
                (lower[0, t_, i], upper[0, t_, i]),
                (lower[1, t_, i], upper[1, t_, i]),
                c="k",
                lw=line_wid,
            )

        ax_posture.plot(
            posture_struct.head_xs, posture_struct.h_y0, "-", c="k", lw=line_wid
        )
        ax_posture.plot(
            posture_struct.head_xs, posture_struct.hy_pos, "-", c="k", lw=line_wid
        )
        ax_posture.plot(
            posture_struct.head_xs, posture_struct.hy_neg, "-", c="k", lw=line_wid
        )
        ax_posture.vlines(
            posture_struct.head_xs,
            posture_struct.hy_neg,
            posture_struct.hy_pos,
            color="k",
            lw=line_wid,
        )
        ax_posture.axes.set_aspect("equal")

        ax_posture.spines[["left", "right", "top", "bottom"]].set_visible(False)
        ax_posture.set(yticks=[], xticks=[], ylim=[-threshold, threshold], xlim=[-3, 1])
        fig.savefig(f"{f_path}/{t_:05}.png", dpi=350)
        plt.close(fig)

    pass


def plot_skeletal_postures(
    trace_data: NDArray,
    lower: NDArray,
    center: NDArray,
    upper: NDArray,
    f_path: str,
    posture_struct: PostureStruct = PostureStruct(),
    fps=700,
    line_wid=2,
):
    f_path = make_dir(f_path)
    trace_data, (tps, n_dims) = orientate_data(trace_data)
    # set frame boundary
    threshold = np.max(np.abs(center[1]))
    time_ms = np.arange(tps) * 1000 / fps
    n_segs = lower.shape[-1]

    for t_ in trange(tps):
        fig, ax_posture = plt.subplots(figsize=(4, 4), dpi=400)
        ax_posture.plot(upper[0, t_, :], upper[1, t_, :], c="k", lw=line_wid)
        ax_posture.plot(center[0, t_, :], center[1, t_, :], c="k", lw=line_wid)
        ax_posture.plot(lower[0, t_, :], lower[1, t_, :], c="k", lw=line_wid)
        for i in range(0, n_segs + 1, (n_segs + 1) // 10):
            ax_posture.plot(
                (lower[0, t_, i], upper[0, t_, i]),
                (lower[1, t_, i], upper[1, t_, i]),
                c="k",
                lw=line_wid,
            )

        ax_posture.plot(
            posture_struct.head_xs, posture_struct.h_y0, "-", c="k", lw=line_wid
        )
        ax_posture.plot(
            posture_struct.head_xs, posture_struct.hy_pos, "-", c="k", lw=line_wid
        )
        ax_posture.plot(
            posture_struct.head_xs, posture_struct.hy_neg, "-", c="k", lw=line_wid
        )
        ax_posture.vlines(
            posture_struct.head_xs,
            posture_struct.hy_neg,
            posture_struct.hy_pos,
            color="k",
            lw=line_wid,
        )
        ax_posture.axes.set_aspect("equal")

        ax_posture.spines[["left", "right", "top", "bottom"]].set_visible(False)
        ax_posture.set(yticks=[], xticks=[], ylim=[-threshold, threshold], xlim=[-3, 1])
        fig.savefig(f"{f_path}/{t_:05}.png", dpi=350)
        plt.close(fig)
    pass


def plot_bout_elapse(
    trace_data: NDArray,
    lower: NDArray,
    center: NDArray,
    upper: NDArray,
    f_path: str,
    line_wid: float = 1,
    t_step: int = 5,
    centre_line: bool = True,
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
    f_path = Path(f_path, "elapse.png")
    # define number of timepoints
    trace_data, (tps, n_dims) = orientate_data(trace_data)
    # make color cycle through time
    threshold = np.max(np.abs(center[1]))
    colors = colormaps["gray"](np.linspace(0.99, 0.05, tps // t_step))

    fig, ax_tail = plt.subplots(figsize=(3, 2), dpi=150)
    ax_tail.set_prop_cycle(cycler(color=colors))
    for t_ in trange(tps - 1, 0, -t_step):
        if not centre_line:
            ax_tail.plot(upper[0, t_, :], upper[1, t_, :], lw=line_wid, alpha=0.6)
            ax_tail.plot(lower[0, t_, :], lower[1, t_, :], lw=line_wid, alpha=0.6)
        ax_tail.plot(center[0, t_, :], center[1, t_, :], lw=line_wid, alpha=0.6)
    ax_tail.spines[["left", "right", "top", "bottom"]].set_visible(False)
    ax_tail.set(yticks=[], xticks=[], ylim=[-threshold, threshold])
    fig.savefig(f_path, transparent=True, dpi=350, bbox_inches="tight")

    # return the colro encoding time
    if return_color_key:
        colors = colormaps["gray"](np.linspace(0.99, 0.05, tps))
        col_key = np.linspace(-np.min(center[0]), np.max(center[0]), tps)
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
            Path(f_path.parent, f"{f_path.stem}_key{f_path.suffix}"),
            transparent=True,
            dpi=350,
            bbox_inches="tight",
        )


if __name__ == "__main__":
    data_arr = np.load(
        "/Users/thomasmullen/VSCodeProjects/fish_simulator/test/fixtures/swim01.npy"
    )
    run(
        data_arr,
        plot_func=plot_skeletal_postures_with_trace,
        dir="/Users/thomasmullen/VSCodeProjects/fish_simulator/dump/plts",
        vid_fp="/Users/thomasmullen/VSCodeProjects/fish_simulator/dump/run_test.mp4",
        line_wid=1,
    )


# def plot_image_and_segments(
#     interpolated_angles: NDArray,
#     head_img: NDArray,
#     seg_imgs: List[NDArray],
#     img_dims: Dict[int, float],
#     i_cntr: int,
# ) -> None:
#     """Plot the head an tail warped by interpolated angles

#     Args:
#         interpolated_angles (NDArray): Interpolated tail angle to transform the img
#         head_img (NDArray): image of fish head
#         seg_imgs (List[NDArray]): list of image segmentes
#         img_dims (Dict[Union[int,float]]): image parameters of dimensions and scale factors
#         i_cntr (int): iterator refering to timepoint of image posture.
#     """
#     fig, ax_tail = plt.subplots(dpi=400)
#     # plot the head
#     ax_tail.imshow(
#         head_img,
#         extent=[
#             0,
#             img_dims["head_x_len"],
#             -img_dims["head_y_len"] // 2,
#             img_dims["head_y_len"] // 2,
#         ],
#         # transform=rot  + ax_tail.transData,
#         transform=ax_tail.transData,
#         alpha=0.8,
#     )
#     # plot segments
#     cum_x_shift = img_dims["head_x_len"]
#     for i, seg in enumerate(seg_imgs):
#         # add a single segment to scale
#         seg_y_len, seg_x_len = seg.shape[:2]
#         ax_tail.imshow(
#             seg,
#             extent=[0, seg_x_len, -seg_y_len // 2, seg_y_len // 2],
#             # transform=trs + rot + ax_tail.transData,
#             transform=transforms.Affine2D().translate(  # Local translation
#                 tx=cum_x_shift, ty=img_dims["head_to_tail_y_offset"]
#             )
#             + transforms.Affine2D().rotate(
#                 np.mod(interpolated_angles[i_cntr, i], 2 * np.pi)
#             )  # Local rotation
#             + ax_tail.transData,  # Global transform
#             alpha=0.8,
#         )
#         cum_x_shift += seg_x_len
#         # image format
#         ax_tail.spines[["left", "right", "top", "bottom"]].set_visible(False)
#         ax_tail.set(
#             yticks=[],
#             xticks=[],
#             xlim=[0, cum_x_shift + 20],
#             ylim=[
#                 (-cum_x_shift) * img_dims["img_sf"],
#                 cum_x_shift * img_dims["img_sf"],
#             ],
#         )
#         plt.close(fig)


# def make_image_simulation(
#     data: NDArray,
#     upsample: int = None,
#     f_path: str = None,
#     img_seg_path: str = None,
# ):
#     """simulate fish dynamics on tail image

#     Args:
#         data (np.ndarray): tail trace
#         upsample (int): Increase the segment resolution and interpolation by dividing by upsample.
#         f_path (str): filepath to save tmp pngs. Default None, will store and delete the temp folder
#         img_seg_path (str): filepath to png segments. Default None, will use default fish

#     Example:
#     >>> tail_traces = np.load("filepath/tail_angle/data.npy")
#     >>> make_image_simulation(data=tail_traces, f_path="/Users/png/dump", upsample=4)
#     >>> make_video(png_dir="/Users/png/dump", vid_fname=path/to/video.mp4, keep_pngs=True)

#     """
#     assert data.ndim == 2, "Need to be 2D"
#     if f_path is None:
#         f_path = tempfile.mkdtemp()
#         print(f"Tmp dir: {f_path}")
#     else:
#         f_path = Path(f_path)
#         f_path.mkdir(parents=True, exist_ok=True)

#     data = data.T if data.shape[0] < data.shape[1] else data
#     tps = data.shape[0]

#     # load image data
#     img_loader = ImageLoader(img_seg_path)
#     # img_loader = ImageLoader("/Users/tom/VSCode/zf_animator_tsm/src/zf_animator_tsm/\
#     #                          template_img/segs")
#     head = img_loader.load_head()
#     segs = img_loader.load_segments()

#     # upsample image
#     if upsample is None:
#         upsample = 1

#     # refine segments
#     segs = list(
#         itertools.chain(*[np.array_split(seg, upsample, axis=1) for seg in segs])
#     )

#     # convert angs to x-y coords
#     tail_x, tail_y = convert_tail_angle_to_keypoints(
#         body_xy=np.zeros((tps, 2)),
#         body_angle=np.zeros(tps),
#         tail_angle=data,
#         body_to_tail_mm=0.5,
#         tail_to_tail_mm=0.32,
#     )
#     # smooth signals
#     intp_x, intp_y = interpolate_keypoints(
#         KeypointStruct(tail_x, tail_y, len(segs) * upsample)
#     )

#     # convert pos to angle
#     intp_angs, body_ang = compute_angles_from_keypoints(
#         body_x=intp_x[:, 0],
#         body_y=intp_y[:, 0],
#         tail_x=intp_x[:, :],
#         tail_y=intp_y[:, :],
#     )

#     # image hyper parameters
#     img_dims = {
#         "head_y_len": head.shape[0],
#         "head_x_len": head.shape[1],
#         "head_to_tail_y_offset": 10,
#         "img_sf": 1.0,
#     }

#     for i_tp in trange(body_ang.size):
#         cum_x_len = img_dims["head_x_len"]
#         # plot the head
#         fig, ax_tail = plt.subplots(dpi=400)
#         ax_tail.imshow(
#             head,
#             extent=[
#                 0,
#                 img_dims["head_x_len"],
#                 -img_dims["head_y_len"] // 2,
#                 img_dims["head_y_len"] // 2,
#             ],
#             # transform=rot  + ax_tail.transData,
#             transform=ax_tail.transData,
#             alpha=0.8,
#         )
#         # plot segments
#         for i, seg in enumerate(segs):
#             # add a single segment to scale
#             seg_y_len, seg_x_len = seg.shape[:2]
#             ax_tail.imshow(
#                 seg,
#                 extent=[0, seg_x_len, -seg_y_len // 2, seg_y_len // 2],
#                 # transform=trs + rot + ax_tail.transData,
#                 transform=transforms.Affine2D().translate(  # Local translation
#                     tx=cum_x_len, ty=img_dims["head_to_tail_y_offset"]
#                 )
#                 + transforms.Affine2D().rotate(
#                     np.mod(intp_angs[i_tp, i], 2 * np.pi)
#                 )  # Local rotation
#                 + ax_tail.transData,  # Global transform
#                 alpha=0.8,
#             )
#             cum_x_len += seg_x_len
#         ax_tail.spines[["left", "right", "top", "bottom"]].set_visible(False)
#         ax_tail.set(
#             yticks=[],
#             xticks=[],
#             xlim=[0, cum_x_len + 20],
#             ylim=[(-cum_x_len) * img_dims["img_sf"], cum_x_len * img_dims["img_sf"]],
#         )

#         fig.savefig(f"{f_path}/{i_tp:05}.png", dpi=150)
#         plt.close(fig)

#         # Contain in a single function - v slow
#         # plot_image_and_segments(
#         #     interpolated_angles=intp_angs,
#         #     head_img=head,
#         #     seg_imgs=segs,
#         #     img_dims=img_dims,
#         #     f_path=f_path,
#         #     i_cntr=i_tp
#         #     )
