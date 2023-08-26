import unittest
import numpy as np
from pathlib import Path
from src.zf_animator_tsm import image_loading


class TestImageLoader(unittest.TestCase):
    def setUp(self):
        print("\nRunning setUp method...")
        self.img_loader_default = image_loading.ImageLoader()
        self.img_loader = image_loading.ImageLoader('./src/zf_animator_tsm/template_img/segs')
    def tearDown(self):
        print("Running tearDown method...")
        
    def test_default_dir_exists(self):
        print("Running test_default_dir_exists")
        print(self.img_loader_default.seg_dir)
        self.assertEqual(Path(self.img_loader.seg_dir).is_dir(), True)
        self.assertEqual(Path(self.img_loader_default.seg_dir).is_dir(), True)
        
    def test_finds_default_dir(self):
        print("Running test_default_dir is true")
        self.assertEqual("src/zf_animator_tsm/template_img/segs" in str(self.img_loader_default.seg_dir), True)
        
    def test_all_segs_loaded(self):
        print("Running test_load_segment images...")
        self.assertEqual(len(self.img_loader.load_segments()), 38)
        self.assertEqual(all(isinstance(seg_, np.ndarray) for seg_ in self.img_loader.load_segments()), True)
        
    def test_head_loaded(self):
        print("Running test_load_head image...")
        self.assertEqual(self.img_loader.load_head().shape, (522, 836, 4))

if __name__=='__main__':
	unittest.main()