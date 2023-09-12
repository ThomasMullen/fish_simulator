"""Locate all relative functions
"""
from .image_loading import ImageLoader
from .tail_transformation import (
    interpolate_keypoints,
    compute_angles_from_keypoints,
    convert_tail_angle_to_keypoints,
    KeypointStruct
)
from .simulator import (
    make_image_simulation,
    make_posture_simulation,
    make_video,
    plot_bout_elapse,
)
