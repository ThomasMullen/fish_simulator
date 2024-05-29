"""Module pull all functionality and generates the images and videos"""
from pathlib import Path
from typing import List, Dict, Tuple, Callable
from tqdm import trange
from numpy.typing import NDArray
import numpy as np
from scipy.interpolate import interp1d
from cycler import cycler
import matplotlib.pyplot as plt
from matplotlib import colormaps
from skimage.transform import PiecewiseAffineTransform, warp

from fish_simulator.transforms import (
    interpolate_keypoints,
    convert_tail_angle_to_keypoints,
    KeypointStruct,
)
from fish_simulator.image_loader import (
    HEAD_IMG,
    TAIL_IMG,
    ImageLoader,
    PostureStruct,
    make_pixel_posture_struct,
)
from fish_simulator.utils import (
    orientate_data,
    make_dir,
    make_video,
    make_color_cycle,
    grey_to_black_cycler,
)


def run(
    data: NDArray,
    plot_func: Callable,
    dir: str,
    vid_fp: str,
    n_intp_segs: int = 49,
    img_kwargs: Dict[float, float] = {
        "body_to_tail_mm": 0.5,
        "tail_to_tail_mm": 0.32,
    },
    **kwargs,
):
    # intp_xy, (low_xy, upp_xy), _ = generate_skeletal_postures(data, *args)
    if plot_func in [plot_tail_image, plot_tail_image_with_trace]:
        posture_struct = make_pixel_posture_struct()
    else:
        posture_struct = PostureStruct()  # or whatever the default is

    intp_xy, (low_xy, upp_xy), _ = generate_skeletal_postures(
        -1 * data, posture_struct, n_intp_segs, img_kwargs=img_kwargs
    )
    plot_func(data, low_xy, intp_xy, upp_xy, dir, **kwargs)
    if vid_fp is not None:
        vid_fp = Path(vid_fp)
        make_video(dir, vid_fp)

    pass


def generate_skeletal_postures(
    data: NDArray,
    posture_struct: PostureStruct = PostureStruct(),
    intp_n_segs: int = 30,
    img_kwargs: Dict[float, float] = {
        "body_to_tail_mm": 0.5,
        "tail_to_tail_mm": 0.32,
    },
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

    onset = (
        0 if img_kwargs["body_to_tail_mm"] == 0.5 else 171
    )  # img_kwargs["body_to_tail_mm"]
    print("onset", onset)

    # convert x-y pos
    tail_x, tail_y = convert_tail_angle_to_keypoints(
        body_xy=np.full((tps, 2), onset),
        body_angle=np.zeros(tps),
        tail_angle=data * -1.2,  # scale factor to account for tail tip
        **img_kwargs,
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


def plot_tail_image(
    trace_data: NDArray,
    lower: NDArray,
    center: NDArray,
    upper: NDArray,
    f_path: str,
    fps=700,
    line_wid=1,
):
    print("Entered plot_tail_image")
    f_path = make_dir(f_path)
    trace_data, (tps, n_dims) = orientate_data(trace_data)
    n_segs = lower.shape[-1]

    y_len, x_len = TAIL_IMG.shape[:2]

    # define base image coords
    top = np.c_[
        np.r_[0, np.arange(0, x_len, x_len // (n_segs - 1))], np.full(n_segs, y_len)
    ]
    mid = np.c_[
        np.r_[0, np.arange(0, x_len, x_len // (n_segs - 1))],
        np.full(n_segs, y_len // 2),
    ]
    bot = np.c_[np.r_[0, np.arange(0, x_len, x_len // (n_segs - 1))], np.zeros(n_segs)]
    src = np.r_[top[:-1], mid[:-1], bot[:-1]]

    for t_ in trange(tps):
        # warped coords
        dst = np.r_[
            lower[:, t_, :].T[:-1], center[:, t_, :].T[:-1], upper[:, t_, :].T[:-1]
        ]
        x_range, y_range = int(np.ceil(np.max(dst[:, 1]))), int(
            np.ceil(np.max(dst[:, 0]))
        )
        # apply transform
        tform = PiecewiseAffineTransform()
        tform.estimate(src[1::3], dst[1::3])
        warped = warp(TAIL_IMG, tform.inverse, output_shape=(x_range, y_range))

        fig, ax_posture = plt.subplots(dpi=200)
        ax_posture.imshow(
            warped,
            cmap=plt.cm.gray,
            origin="lower",
        )
        ax_posture.imshow(HEAD_IMG, cmap=plt.cm.gray, extent=[-709, 40, 0, 342])
        ax_posture.set(
            xlim=(-709, x_len),
            ylim=(-x_len, x_len),
            yticks=[],
            xticks=[],
        )
        # ax_posture.spines[["left", "right", "top", "bottom"]].set_visible(False)
        ax_posture.axis("off")

        fig.tight_layout()
        fig.savefig(f"{f_path}/{t_:05}.png", dpi=150)
        plt.close(fig)
    pass


def plot_tail_image_with_trace(
    trace_data: NDArray,
    lower: NDArray,
    center: NDArray,
    upper: NDArray,
    f_path: str,
    fps=700,
    line_wid=1,
):
    print("Entered plot_tail_image")
    f_path = make_dir(f_path)
    trace_data, (tps, n_dims) = orientate_data(trace_data)
    time_ms = np.arange(tps) * 1000 / fps
    n_segs = lower.shape[-1]
    y_len, x_len = TAIL_IMG.shape[:2]

    # define base image coords
    top = np.c_[
        np.r_[0, np.arange(0, x_len, x_len // (n_segs - 1))], np.full(n_segs, y_len)
    ]
    mid = np.c_[
        np.r_[0, np.arange(0, x_len, x_len // (n_segs - 1))],
        np.full(n_segs, y_len // 2),
    ]
    bot = np.c_[np.r_[0, np.arange(0, x_len, x_len // (n_segs - 1))], np.zeros(n_segs)]
    src = np.r_[top[:-1], mid[:-1], bot[:-1]]

    for t_ in trange(tps):
        # warped coords
        dst = np.r_[
            lower[:, t_, :].T[:-1], center[:, t_, :].T[:-1], upper[:, t_, :].T[:-1]
        ]
        x_range, y_range = int(np.ceil(np.max(dst[:, 1]))), int(
            np.ceil(np.max(dst[:, 0]))
        )
        # apply transform
        tform = PiecewiseAffineTransform()
        tform.estimate(src[1::3], dst[1::3])
        warped = warp(TAIL_IMG, tform.inverse, output_shape=(x_range, y_range))

        fig, (ax_trace, ax_posture) = plt.subplots(1, 2, figsize=(8, 4), dpi=200)
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
        ax_posture.imshow(
            warped,
            cmap=plt.cm.gray,
            origin="lower",
        )
        ax_posture.imshow(HEAD_IMG, cmap=plt.cm.gray, extent=[-709, 40, 0, 342])
        ax_posture.set(
            xlim=(-709, x_len),
            ylim=(-x_len, x_len),
            yticks=[],
            xticks=[],
        )
        ax_posture.axis("off")

        fig.tight_layout()
        fig.savefig(f"{f_path}/{t_:05}.png", dpi=150)
        plt.close(fig)
    pass


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

        # ax_posture.spines[["left", "right", "top", "bottom"]].set_visible(False)
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
    data_name = "swim01"
    data_arr = np.load(
        f"/Users/thomasmullen/VSCodeProjects/fish_simulator/test/fixtures/{data_name}.npy"
    )
    # test posture animation
    run(
        data_arr,
        plot_func=plot_skeletal_postures_with_trace,
        dir="/Users/thomasmullen/VSCodeProjects/fish_simulator/dump/plts",
        vid_fp=f"/Users/thomasmullen/VSCodeProjects/fish_simulator/dump/run_test_{data_name}.mp4",
        n_intp_segs=30,
        line_wid=1,
    )
    # test tail animation
    run(
        data_arr,
        plot_func=plot_tail_image_with_trace,
        # plot_func=plot_tail_image,
        dir="/Users/thomasmullen/VSCodeProjects/fish_simulator/dump/plts",
        vid_fp=f"/Users/thomasmullen/VSCodeProjects/fish_simulator/dump/run_test_tail_trace_{data_name}.mp4",
        n_intp_segs=40,
        img_kwargs={"body_to_tail_mm": 156.3, "tail_to_tail_mm": -181.3},
        line_wid=1,
    )
