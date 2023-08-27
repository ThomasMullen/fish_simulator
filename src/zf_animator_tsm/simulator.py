import os
import itertools
from pathlib import Path
from tqdm import trange
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib import transforms

from .image_loading import ImageLoader
from .tail_transformation import (
    interpolate_keypoints,
    compute_angles_from_keypoints,
    convert_tail_angle_to_keypoints,
    KeypointStruct,
)


def make_simulation(
    data: np.ndarray,
    vid_fname: str,
    upsample: int = None,
    fp: str = None,
    keep_pngs: bool = False,
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
    if fp is None:
        import tempfile

        fp = tempfile.mkdtemp()
        print("Tmp dir: {}".format(fp))
    else:
        fp = Path(fp)
        fp.mkdir(parents=True, exist_ok=True)

    data = data.T if data.shape[0] < data.shape[1] else data
    tps, n_seg = data.shape

    # load image data
    img_loader = ImageLoader()
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
        tail_x, tail_y, n_segments=KeypointStruct(tail_x, tail_y, len(segs) * upsample)
    )

    # convert pos to angle
    intp_angs, body_ang = compute_angles_from_keypoints(
        body_x=intp_x[:, 0],
        body_y=intp_y[:, 0],
        tail_x=intp_x[:, :],
        tail_y=intp_y[:, :],
    )

    # image hyper parameters
    head_y_len, head_x_len = head.shape[:2]
    y_offset = 10
    sf = 1.0

    for t_ in trange(body_ang.size):
        cum_x_len = head_x_len
        # plot the head
        fig, ax = plt.subplots(dpi=400)
        rot = transforms.Affine2D().rotate(np.mod(body_ang[t_], 2 * np.pi))
        ax.imshow(
            head,
            extent=[0, head_x_len, -head_y_len // 2, head_y_len // 2],
            # transform=rot  + ax.transData,
            transform=ax.transData,
            alpha=0.8,
        )
        # plot segments
        for i, seg in enumerate(segs):
            # add a single segment to scale
            seg_y_len, seg_x_len = seg.shape[:2]
            rot = transforms.Affine2D().rotate(np.mod(intp_angs[t_, i], 2 * np.pi))
            trs = transforms.Affine2D().translate(tx=cum_x_len, ty=y_offset)
            ax.imshow(
                seg,
                extent=[0, seg_x_len, -seg_y_len // 2, seg_y_len // 2],
                transform=trs + rot + ax.transData,
                alpha=0.8,
            )
            cum_x_len += seg_x_len
        ax.spines[["left", "right", "top", "bottom"]].set_visible(False)
        ax.set(
            yticks=[],
            xticks=[],
            xlim=[0, cum_x_len + 20],
            # ylim=[-head_y_len//2, head_y_len//2],
            # ylim=[-head_y_len*sf, head_y_len*sf],
            ylim=[(-cum_x_len) * sf, cum_x_len * sf],
            # ylim=[(-cum_x_len+head_x_len)*sf, cum_x_len-head_x_len*sf],
        )

        fig.savefig(f"{fp}/{t_:03}.png", dpi=150)
        plt.close(fig)

    if os.path.exists(vid_fname):
        os.remove(vid_fname)

    # make video
    cmd = "ffmpeg -r 35 -f image2 -i {}/%03d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p {}".format(
        fp, vid_fname
    )
    os.system(cmd)
    print("Saving video to: {}".format(vid_fname))

    if not keep_pngs:
        os.remove(fp)
        print("delete: {} folder".format(fp))
