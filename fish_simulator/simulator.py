"""Module pull all functionality and generates the images and videos"""
import os
import itertools
from pathlib import Path
import tempfile
from typing import List, Dict
from tqdm import trange
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib import transforms


from fish_simulator.tail_transformation import (
    interpolate_keypoints,
    compute_angles_from_keypoints,
    convert_tail_angle_to_keypoints,
    KeypointStruct,
)
from fish_simulator.image_loading import ImageLoader


def make_simulation(
    data: NDArray,
    upsample: int = None,
    f_path: str = None,
):
    """simulate fish dynamics on tail image

    Args:
        data (np.ndarray): tail trace
        fp (str): filepath to save tmp pngs. Default None, will store and delete the temp folder.
        vid_fname (str): video filepath to store
        image_tmp (dict): dict('head', 'segs') containing image parts as np arrays

    Example:
    >>> head = np.load('./segs.npy', allow_pickle=True)
    >>> segs = np.load('./head.npy')
    >>> image_tmp = dict(head=head, segs=segs)
    >>> make_simulation(data=traces, fp="/Users/dump",
    vid_fname"/Users/dump/vid.mp4, image_tmp=image_tmp)

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
    img_loader = ImageLoader()
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

        # Contain in a single function
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


def make_video(png_dir: str, vid_fname: str, keep_pngs: bool = True) -> None:
    """converts the save png figs to mp4 with ffmpeg

    Args:
        png_dir (str): directory with numbered pngs
        vid_fname (str): video filepath
        keep_pngs (bool, optional): delete dir with png after video saved.
        Defaults to True.
    """
    if os.path.exists(vid_fname):
        os.remove(vid_fname)

    # make video
    cmd = f"ffmpeg -r 35 -f image2 -i {png_dir}/%03d.png -vcodec libx264 \
    -crf 25 -pix_fmt yuv420p {vid_fname}"
    os.system(cmd)
    print(f"Saving video to: {vid_fname}")

    if not keep_pngs:
        try:
            os.remove(png_dir)
            print(f"delete: {png_dir} folder")
        except:
            print(f"Not permitted to delete dir:\n{png_dir}")
