import numpy as np
import os
from PIL import Image
from pathlib import Path
from typing import List, Dict
from numpy.typing import NDArray


class ImageLoader:
    def __init__(self, dir=None) -> None:
        self.seg_dir = Path(os.getcwd(), "src", "zf_animator_tsm", "template_img", "segs") if dir is None else dir
        self.head_dir = Path(self.seg_dir).parent
        print(self.seg_dir)
        
    def load_segments(self) -> List[NDArray]:
        n_segs = len(list(Path(self.seg_dir).iterdir()))
        segs = [np.asarray(Image.open(f"{self.seg_dir}/{i:02}.png")) for i in range(n_segs)]
        return segs
        
    def load_head(self) -> np.ndarray:
        head = np.asarray(Image.open(f"{self.head_dir}/fish_head.png"))
        return head
        
    def return_zf_img(self) -> Dict:
        zf_image = dict(
                        head=self.load_head(),
                        segs=self.load_segments(),
                    )
        return zf_image
    