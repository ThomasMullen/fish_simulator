"""Unit test ImageLoader class functionality"""
import unittest
from pathlib import Path
import numpy as np
from fish_simulator import image_loading


class TestImageLoader(unittest.TestCase):
    """Import class tester for ImageLoader"""
    def setUp(self):
        """Instantiate default and constructed class
        """
        print("\nRunning setUp method...")
        self.img_loader_default = image_loading.ImageLoader()
        self.img_loader = image_loading.ImageLoader(
            "./fish_simulator/template_img/segs"
        )

    def tearDown(self):
        """Destruct test class
        """
        print("Running tearDown method...")

    def test_default_dir_exists(self):
        """Test if directories exist in repo
        """
        print("Running test_default_dir_exists")
        print(self.img_loader_default.seg_dir)
        self.assertEqual(Path(self.img_loader.seg_dir).is_dir(), True)
        self.assertEqual(Path(self.img_loader_default.seg_dir).is_dir(), True)

    def test_finds_default_dir(self):
        """Test the default directory is valid
        """
        print("Running test_default_dir is true")
        self.assertEqual(
            "fish_simulator/template_img/segs"
            in str(self.img_loader_default.seg_dir),
            True,
        )

    def test_all_segs_loaded(self):
        """Check dimensions of segments loaded and dtype within list
        """
        print("Running test_load_segment images...")
        self.assertEqual(len(self.img_loader.load_segments()), 38)
        self.assertEqual(
            all(
                isinstance(seg_, np.ndarray) for seg_ in self.img_loader.load_segments()
            ),
            True,
        )

    def test_head_loaded(self):
        """Test dimensions of loaded head
        """
        print("Running test_load_head image...")
        self.assertEqual(self.img_loader.load_head().shape, (522, 836, 4))


if __name__ == "__main__":
    unittest.main()
