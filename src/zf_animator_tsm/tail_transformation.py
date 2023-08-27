"""Module that contains transformations from tail angle
to x-y coorindates. Additionally, with interpolation functions
"""
from typing import Tuple
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import interp1d
from numpy.typing import NDArray


@dataclass
class KeypointStruct:
    """Structure for key x-y- points timeseries along tail (Each 2D [TxN]),
    and number of tail segs

    Raises:
        ValueError: Insufficient number of keypoints to interpolate.
    """

    tail_x: NDArray
    tail_y: NDArray
    n_segments: int

    def __post_init__(self):
        if self.n_segments < 2:
            raise ValueError("There should be more than 3 keypoints.")


def interpolate_keypoints(key_pnt_struc: KeypointStruct) -> Tuple[NDArray, NDArray]:
    """Interpolates the tail keypoints to create a curve with a specified
    number of segments. via cubis or linear using tail constructor.

    Returns:
        Tuple: Interpolated x and y keypoints
    """
    n_tps, n_segments_init = (
        key_pnt_struc.tail_x.shape[0],
        key_pnt_struc.tail_x.shape[1],
    )
    tail_x_interp = np.zeros((n_tps, key_pnt_struc.n_segments + 1)) * np.nan
    tail_y_interp = np.zeros((n_tps, key_pnt_struc.n_segments + 1)) * np.nan

    for i_tp in range(n_tps):
        # try:
        points = np.array(
            [key_pnt_struc.tail_x[i_tp, :], key_pnt_struc.tail_y[i_tp, :]]
        ).T
        is_nan = np.any(np.isnan(points))

        if not is_nan:
            id_first_nan = points.shape[0]
            n_segments = key_pnt_struc.n_segments + 1
        else:
            id_first_nan = np.where(np.any(np.isnan(points), axis=1))[0][0]
            n_segments = int(
                np.round(
                    id_first_nan / n_segments_init * (key_pnt_struc.n_segments + 1)
                )
            )

        alpha = np.linspace(0, 1, n_segments)
        distance = np.cumsum(
            np.sqrt(np.sum(np.diff(points[:id_first_nan, :], axis=0) ** 2, axis=1))
        )
        distance = np.insert(distance, 0, 0) / distance[-1]

        if len(distance) > 3:
            interpolator = interp1d(
                distance, points[:id_first_nan, :], kind="cubic", axis=0
            )
        else:
            interpolator = interp1d(
                distance, points[:id_first_nan, :], kind="linear", axis=0
            )

        curve = interpolator(alpha)

        tail_x_interp[i_tp, : key_pnt_struc.n_segments+1] = curve[:, 0]
        tail_y_interp[i_tp, : key_pnt_struc.n_segments+1] = curve[:, 1]
        # except Exception as e:
        #     print(f"Error {e} occurred.")
        #     print(f"Keypoint interpolation failed tp: {i_tp}")

    return tail_x_interp, tail_y_interp


def convert_tail_angle_to_keypoints(
    body_xy: NDArray,
    body_angle: NDArray,
    tail_angle: NDArray,
    body_to_tail_mm: float = 0.5,
    tail_to_tail_mm: float = 0.32,
) -> Tuple[NDArray, NDArray]:
    """Convert tail angle segments to x-y positions

    Args:
        body_xy (NDArray): 2D array of body xy-position of length time,
        shape: [time x 2].
        body_angle (NDArray): 1D array of body angle of length time
        tail_angle (NDArray): 2D array tail angle data, shape (time x n_segments)
        body_to_tail_mm (float, optional): Separation between tail and body point. Defaults to 0.5.
        tail_to_tail_mm (float, optional): Separation between each measured tail angle.
        Defaults to 0.32.

    Raises:
        ValueError: time length of body x-pos array different to tail angle time length, or
        time length of body y-pos array different to tail angle time length, or
        time length of body angle array different to tail angle time length.

    Notes:
        Shape should be body_xy = np.zeros((n_tps,2)); body_angle = np.zeros(n_tps)

    Returns:
        Tuple[NDArray, NDArray]: 2D array converted measured tail angle x-position through time, and
        y-position through time, both of shape (time x n_segments)
    """
    n_tps, n_tail_segments = tail_angle.shape
    if (body_xy.shape[0] != n_tps) or (len(body_angle) != n_tps):
        raise ValueError("incompatible dimensions")

    tail_xy = np.zeros((n_tps, n_tail_segments + 1, 2))

    for i_tp in range(n_tps):
        # x, y = body_x[t], body_y[t]
        # head_pos = np.array([body_xy[i_tp]])
        body_vect = np.array([np.cos(body_angle[i_tp]), np.sin(body_angle[i_tp])])

        swim_bladder = body_xy[i_tp] - body_vect * body_to_tail_mm

        tail_xy[i_tp, 0] = swim_bladder
        tail_angle_abs = tail_angle[i_tp, :] + (body_angle[i_tp] + np.pi)
        tail_pos = np.copy(swim_bladder)
        for j_seg in range(n_tail_segments):
            tail_vect = np.array(
                [np.cos(tail_angle_abs[j_seg]), np.sin(tail_angle_abs[j_seg])]
            )
            tail_pos += tail_to_tail_mm * tail_vect
            tail_xy[i_tp, j_seg + 1] = tail_pos

    return np.squeeze(np.split(tail_xy[:, :], 2, axis=2))


def interpolate_tail_angle(
    tail_angle: NDArray, n_segments: int = 10
) -> Tuple[NDArray, NDArray]:
    """Converts measured tail angles to keypoints, applies keypoint interpolation,
    then computes angles from inpterpolated keypoints.

    Args:
        tail_angle (NDArray): 2D array in shape (time x n_segments)
        n_segments (int, optional): Number of segments want to interpolate along tail.
        Defaults to 10.

    Returns:
        Tuple[NDArray, NDArray]: returns the interpolated tail angles (2D array), and body angle
        (1D array) through time.
    """
    n_tps = tail_angle.shape[0]
    # Convert to keypoints
    body_xy = np.zeros((n_tps, 2))
    body_angle = np.zeros(n_tps)
    tail_x, tail_y = convert_tail_angle_to_keypoints(body_xy, body_angle, tail_angle)

    # Interpolate
    tail_x_interp, tail_y_interp = interpolate_keypoints(
        KeypointStruct(tail_x, tail_y, n_segments)
    )

    # Compute tail angle
    tail_angle_interp, body_angle_interp = compute_angles_from_keypoints(
        body_xy[:, 0], body_xy[:1], tail_x_interp, tail_y_interp
    )

    return tail_angle_interp, body_angle_interp


def compute_angle_between_vectors(vec_1, vec_2):
    """
    Computes the angle between two vectors.

    Args:
        vec_1 (ndarray): First set of vectors with shape (num_vectors, num_dimensions).
        vec_2 (ndarray): Second set of vectors with shape (num_vectors, num_dimensions).

    Returns:
        ndarray: Array of angles between the vectors.
    """
    dot_product = np.einsum("ij,ij->i", vec_1, vec_2)
    # norms_product = np.linalg.norm(vec_1, axis=1) * np.linalg.norm(vec_2, axis=1)
    cos_angle = dot_product
    sin_angle = np.cross(vec_1, vec_2)
    angle = np.arctan2(sin_angle, cos_angle)

    return angle


def compute_angles_from_keypoints(
    body_x: NDArray, body_y: NDArray, tail_x: NDArray, tail_y: NDArray
) -> Tuple[NDArray, NDArray]:
    """
    Computes the tail angles and body angle based on keypoints.

    Args:
        body_x (ndarray): X-coordinates of the body keypoints.
        body_y (ndarray): Y-coordinates of the body keypoints.
        tail_x (ndarray): X-coordinates of the tail keypoints.
        tail_y (ndarray): Y-coordinates of the tail keypoints. .
    Returns:
        tuple: A tuple containing the tail angles (tail_angle) and the body angle (body_angle).

    Raises:
        ValueError: If tail_x and tail_y do not have the expected shape or if the time axis
        is different.
    """
    n_keypoints = tail_x.shape[1]
    if tail_x.shape != tail_y.shape:
        raise ValueError("tail_x and tail_y must have same dimensions")

    n_tps = body_x.shape[0]
    vector_tail_segment = np.concatenate(
        (
            np.diff(tail_x, axis=1)[:, :, np.newaxis],
            np.diff(tail_y, axis=1)[:, :, np.newaxis],
        ),
        axis=2,
    )
    vector_tail_segment = np.swapaxes(vector_tail_segment, 1, 2)
    start_vector = np.vstack((tail_x[:, 0] - body_x, tail_y[:, 0] - body_y)).T
    body_vector = -start_vector
    body_angle = np.arctan2(body_vector[:, 1], body_vector[:, 0])
    body_angle[~np.isnan(body_angle)] = np.unwrap(body_angle[~np.isnan(body_angle)])
    relative_angle = np.zeros((n_tps, n_keypoints - 1))
    print(vector_tail_segment.shape)
    for k_pnt in range(n_keypoints - 1):
        relative_angle[:, k_pnt] = compute_angle_between_vectors(
            start_vector, vector_tail_segment[:, :, k_pnt]
        )
        start_vector = np.copy(vector_tail_segment[:, :, k_pnt])
    tail_angle = np.cumsum(relative_angle, 1)

    return tail_angle, body_angle
