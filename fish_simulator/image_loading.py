"""Module loading image for animation."""
import os
from pathlib import Path
from typing import List, Dict
import numpy as np
from numpy.typing import NDArray
from PIL import Image


class ImageLoader:
    """Self contained loader that returns the tail segments and head images."""

    def __init__(self, seg_dir=None) -> None:
        self.seg_dir = (
            # Path(os.getcwd(), "src", "fish_simulator", "template_img", "segs")
            Path(os.getcwd(), "fish_simulator", "template_img", "segs")
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
        n_segs = len(list(Path(self.seg_dir).iterdir()))
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
