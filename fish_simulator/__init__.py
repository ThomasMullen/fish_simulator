"""Locate all relative functions
"""
from .image_loading import ImageLoader
from .tail_transformation import (
    interpolate_keypoints,
    compute_angles_from_keypoints,
    convert_tail_angle_to_keypoints,
    KeypointStruct
)
from .simulator import make_simulation
