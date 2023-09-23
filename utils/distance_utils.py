import numpy as np


def transform_to_formula_angle(angle: float) -> float:
    return np.radians(angle)


def get_point_on_ground_plane(p1, p2, camera_h, angle):
    a = np.sin(angle)
    b = np.cos(angle)

    lam = -camera_h * (a ** 2 + b ** 2) / (a + b * p2[1])
    _px = p2[0] * lam
    _py = -camera_h
    _pz = camera_h * (a * p2[1] - b) / (a + b * p2[1])
    return np.array([_px, _py, _pz], dtype=np.float32)


def get_two_points_from_pixel_projection(pix_coord, camera_K):
    assert len(pix_coord) == 2
    iK = np.linalg.inv(camera_K)
    glob_point = iK @ np.array([*pix_coord, 1])
    glob_point = glob_point / glob_point[2]
    glob_point1 = np.array([0, 0, 0], dtype=np.float32)
    glob_point2 = glob_point.copy()

    return glob_point1, glob_point2


def warp_point_to_XZ(pix_coord, camera_h, camera_K, alpha):
    line_point_1, line_point_2 = get_two_points_from_pixel_projection(
        pix_coord, camera_K
    )
    ground_intersection_point = get_point_on_ground_plane(
        line_point_1, line_point_2, -camera_h, alpha
    )
    return ground_intersection_point[0], ground_intersection_point[2]


def get_distance_to_point(pix_coord, camera_h, camera_K, alpha):
    distance_to_point = np.linalg.norm(
        warp_point_to_XZ(pix_coord, camera_h, camera_K, alpha)
    )
    return distance_to_point


def find_v_for_u(u: int, d: float, camera_h, inv_camera_K, image_h, angle):
    ifx = inv_camera_K[0, 0]
    ify = inv_camera_K[1, 1]
    icx = inv_camera_K[0, 2]
    icy = inv_camera_K[1, 2]
    a = np.sin(angle)
    b = np.cos(angle)

    x2 = u * ifx + icx
    p = 2 * a * b * (d ** 2 - camera_h ** 2) / (b ** 2 * d ** 2 - a ** 2 * camera_h ** 2)
    q = (a ** 2 * d ** 2 - b ** 2 * camera_h ** 2 - x2 ** 2 * camera_h ** 2 * ((a ** 2 + b ** 2) ** 2)) / \
        (b ** 2 * d ** 2 - a ** 2 * camera_h ** 2)

    di2 = p ** 2 - 4 * q
    if di2 < 1E-5:
        raise RuntimeWarning('Unvisible position, camera too low or too hight')

    y2 = (-p + np.sqrt(di2)) / 2

    v = (y2 - icy) / ify

    if v >= image_h:
        raise RuntimeWarning('Unvisible position, camera too low')

    return v


def calcle_distance_line(d: float, image_shape: tuple,
                         camera_h: float, camera_K: np.ndarray, angle: float) -> np.ndarray:
    assert d > 0
    h, w = image_shape

    line_pixels = np.zeros(w, dtype=np.int32)
    iK = np.linalg.inv(camera_K)

    for px in range(w):
        try:
            py = find_v_for_u(px, d, camera_h, iK, h, angle)
        except:
            line_pixels[px] = image_shape[0] - 1
        else:
            line_pixels[px] = py

    return line_pixels


def get_intrinsics(FOV, H, W, c_factor: float = 0.5):
    """
    Intrinsics for a pinhole camera model.
    """
    fx = c_factor * W / np.tan(0.5 * FOV * np.pi / 180.0)
    fy = c_factor * H / np.tan(0.5 * FOV * np.pi / 180.0)
    cx = 0.5 * W
    cy = 0.5 * H
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]])


def fov_to_focal_length(FOV, W):
    return W / 2 / np.tan(FOV * np.pi / 180.0 / 2)
