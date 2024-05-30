"""Module loading image for animation."""
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray
from PIL import Image

TAIL_PATH = Path(Path(__file__).parent, "templates", "template_img", "segs", "tail.png")
HEAD_PATH = Path(Path(__file__).parent, "templates", "template_img", "head.png")
TAIL_IMG = np.flipud(np.asarray(Image.open(TAIL_PATH)))
HEAD_IMG = np.asarray(Image.open(HEAD_PATH))


class ImageLoader:
    """Self contained loader that returns the tail segments and head images."""

    def __init__(self, seg_dir=None) -> None:
        self.seg_dir = (
            Path(Path(__file__).parent, "templates", "template_img2", "segs")
            if seg_dir is None
            else seg_dir
        )
        self.head_dir = Path(self.seg_dir).parent
        print(self.seg_dir)

    def load_segments(self) -> List[NDArray]:
        """Load list of segment pngs into numpy arrays of different shape

        Returns:
            List[NDArray]: elements of numpy arrays of different shape forming tail.
        """
        n_segs = len(list(Path(self.seg_dir).rglob("*.png")))
        segs = [
            np.asarray(Image.open(f"{self.seg_dir}/{i:02}.png")) for i in range(n_segs)
        ]
        return segs

    def load_head(self) -> np.ndarray:
        """Load the head of fish

        Returns:
            np.ndarray: head 2D image array
        """
        head = np.asarray(Image.open(f"{self.head_dir}/fish_head.png"))
        return head

    def return_zf_img(self) -> Dict:
        """Merge tail and head dataset to a dictionary

        Returns:
            Dict: keys 'head' and 'segs' composing tail.
        """
        zf_image = {"head": self.load_head(), "segs": self.load_segments()}
        return zf_image


@dataclass
class PostureStruct:
    seg_width: np.ndarray = field(
        default_factory=lambda: np.array(
            [3.871, 3.517, 3.192, 2.772, 2.394, 1.832, 1.3, 0.0]
        )
        * 0.05085
    )
    head_width: np.ndarray = field(
        default_factory=lambda: np.array(
            [3.871, 4.436, 5.739, 5.157, 8.651, 10.481, 10.481, 11.867, 9.982, 4.547]
        )
        * 0.05085
    )
    head_len: float = 25.90 * 0.05085
    body_to_tail = 0.5
    head_xs: np.ndarray = field(
        default_factory=lambda: np.linspace(0, 25.90 * 0.05085, 10) - 0.5
    )
    h_y0: np.ndarray = field(default_factory=lambda: np.full(10, 0))
    hy_pos: np.ndarray = field(
        default_factory=lambda: (
            np.array(
                [
                    3.871,
                    4.436,
                    5.739,
                    5.157,
                    8.651,
                    10.481,
                    10.481,
                    11.867,
                    9.982,
                    4.547,
                ]
            )
        )
        * 0.05085
        / 2
    )
    hy_neg: np.ndarray = field(
        default_factory=lambda: -(
            np.array(
                [
                    3.871,
                    4.436,
                    5.739,
                    5.157,
                    8.651,
                    10.481,
                    10.481,
                    11.867,
                    9.982,
                    4.547,
                ]
            )
        )
        * 0.05085
        / 2
    )
    n_segs: int = 7
    xs: np.ndarray = field(
        default_factory=lambda: np.linspace(0, 49.48 * 0.05085, 7 + 1)
    )
    y_0: np.ndarray = field(default_factory=lambda: np.full(7 + 1, 0))
    y_pos: np.ndarray = field(
        default_factory=lambda: (
            np.array([3.871, 3.517, 3.192, 2.772, 2.394, 1.832, 1.3, 0.0]) * 0.05085
        )
        / 2
    )
    y_neg: np.ndarray = field(
        default_factory=lambda: -(
            np.array([3.871, 3.517, 3.192, 2.772, 2.394, 1.832, 1.3, 0.0]) * 0.05085
        )
        / 2
    )


def make_pixel_posture_struct() -> PostureStruct:
    """Generate the posture struct for pixel posture

    Returns:
        PostureStruct: adapted default posture struct for pixel posture
    """
    img_wid, img_height = TAIL_IMG.shape[:2]
    xs = np.linspace(0, img_height, 7)
    y_pos = np.array([img_wid] * 8)
    y_0 = np.array([img_wid // 2] * 8)
    y_neg = np.array([0] * 8)
    seg_width = np.array([img_wid] * 8)
    return PostureStruct(xs=xs, y_pos=y_pos, y_neg=y_neg, y_0=y_0, seg_width=seg_width)
