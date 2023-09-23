from typing import Callable, Optional, Tuple, List, Dict
import cv2
import numpy as np
import os


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


def search_closest_coordinate(height: int, x: int, target: float, dist_func: Callable) -> Optional[int]:
    low = 0
    high = height - 1

    while low <= high:
        mid = (low + high) // 2
        current_number = dist_func((x, mid))

        if current_number == target:
            return mid
        elif current_number > target:
            low = mid + 1
        else:
            high = mid - 1

    low = height if low > height else low
    high = 0 if high < 0 else high

    left_difference = abs(dist_func((x, high)) - target)
    right_difference = abs(dist_func((x, low)) - target)

    if left_difference <= right_difference and left_difference <= 0.01:
        return high

    if right_difference <= 0.01:
        return low

    return None


def render_dist_frame(image, train_asset, matrix, warp_size, dets: list, dist_info=None, lines_asset=None):
    wout = cv2.warpPerspective(
        image.copy(), matrix, warp_size, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT
    )

    padded = np.pad(
        wout,
        (
            (0, train_asset.shape[0] - int(train_asset.shape[0] * 0.2)),
            (0, 0),
            (0, 0)
        ),
        # 'linear_ramp',
        'constant',
        constant_values=0
    )
    y1 = wout.shape[0] - int(train_asset.shape[0] * 0.2)
    x1 = wout.shape[1] // 2 - train_asset.shape[1] // 2
    padded[
    y1:y1 + train_asset.shape[0], x1:x1 + train_asset.shape[1]
    ][train_asset[..., 0] > 0] = train_asset[..., :3][train_asset[..., 0] > 0]

    if dist_info is not None or lines_asset is not None:
        if lines_asset is None:
            lines_asset = np.zeros((padded.shape[0], padded.shape[1], 4), dtype=np.uint8)

            lines_colors = ((10, 20, 200, 255), (10, 200, 200, 255), (10, 200, 20, 255))
            text_font = cv2.FONT_HERSHEY_DUPLEX
            text_scale = 3.0
            text_thickness = 7

            for line_i, line_d in enumerate(dist_info['distances']):
                line = calcle_distance_line(
                    line_d,
                    image.shape[:2],
                    dist_info['h'],
                    dist_info['K'],
                    dist_info['angle']
                )
                line_points = np.array([[x, line[x]] for x in range(len(line))])
                line_pts = line_points.astype(np.float32)[:, None]
                warped_line_pts = cv2.perspectiveTransform(line_pts, matrix).squeeze(1)

                warped_full_x = np.arange(warped_line_pts[:, 0].min(), warped_line_pts[:, 0].max(), 1)
                yinterp = np.interp(
                    warped_full_x,
                    warped_line_pts[:, 0], warped_line_pts[:, 1]
                )
                warped_line_pts = np.stack((warped_full_x, yinterp), axis=1)

                for i, j in warped_line_pts.tolist():
                    lines_asset = cv2.circle(lines_asset, (int(i), int(j)), 9, lines_colors[line_i], -1)

                px_step = 200
                lines_asset = cv2.line(
                    lines_asset,
                    (50, 2400 - line_i * px_step), (300, 2400 - line_i * px_step),
                    lines_colors[line_i], 31
                )

                lines_asset = cv2.putText(
                    lines_asset,
                    str(line_d) + ' ' * (3 - len(str(line_d))) + ' m',
                    (350, 2400 - line_i * px_step + 20),
                    text_font,
                    text_scale,
                    lines_colors[line_i],
                    text_thickness
                )

        padded[lines_asset[..., 3] > 0] = lines_asset[..., :3][lines_asset[..., 3] > 0].copy()
        # lines_asset = cv2.cvtColor(lines_asset, cv2.COLOR_RGBA2BGRA)

    if len(dets) > 0:
        pts = []
        for det in dets:
            x1, y1, x2, y2 = det[0]
            xc = (x2 + x1) / 2
            yc = y2
            pts.append([xc, yc])

        pts = np.array(pts).astype(np.float32)[:, None]
        warped_pts = cv2.perspectiveTransform(pts, matrix).squeeze(1)

        for p in warped_pts:
            padded = cv2.circle(padded, (int(p[0]), int(p[1])), 30, (20, 200, 50), 15)

    return padded, lines_asset


def draw_lines(image, matrix, angle=2.5, regions=[10, 20, 100], camera_h: float= 2, swapRB: bool=False, non_warped_lines=None):
    assert len(regions) == 3
    res = image

    if non_warped_lines is None:
        non_warped_lines = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)

        _angle = transform_to_formula_angle(angle)

        r, y, g = (200, 20, 10), (200, 200, 10), (20, 200, 10)

        if swapRB:
            r = r[::-1]
            y = y[::-1]
            g = g[::-1]

        for line_color, d in zip((r, y, g), regions):
            dl = calcle_distance_line(d, image.shape[:2], camera_h, matrix, _angle)
            for i, j in enumerate(dl):
                non_warped_lines = cv2.circle(
                    non_warped_lines, (i, j), 3,
                    (line_color[0], line_color[1], line_color[2], 255),
                    -1
                )

    res[non_warped_lines[..., 3] > 0] = non_warped_lines[..., :3][non_warped_lines[..., 3] > 0].copy()

    return res, non_warped_lines


def merge_frames(img1, img2, split_line_width: int = 20):
    fy = img1.shape[0] / img2.shape[0]

    new_size2 = (int(img2.shape[1] * fy), int(img2.shape[0] * fy))
    new_img2 = cv2.resize(img2, new_size2, interpolation=cv2.INTER_NEAREST)
    sp_line = np.zeros((new_size2[1], split_line_width, 3), dtype=img1.dtype)
    return np.concatenate((img1, sp_line, new_img2), axis=1)


def gen_warp_matrix(image_shape: Tuple[int, int]) -> Tuple[np.ndarray, Tuple[int, int]]:
    dst_points = [
        [0, 0],
        [image_shape[1] - 1, 0],
        [image_shape[1] - 1, image_shape[0] * 4 - 1],
        [0, image_shape[0] * 4 - 1],
    ]

    d1 = 135
    d2 = 100

    shift_of_view = 2 * image_shape[0] // 3

    src_points = [
        [960 - d1, 518],
        [960 + d1, 518],
        [1920 - d2, 830],
        [d2, 830]
    ]

    dst_points = np.array(dst_points, np.float32) / 2
    src_points = np.array(src_points, np.float32)

    dst_points[:, 0] += shift_of_view

    mat = cv2.getPerspectiveTransform(src_points, dst_points)
    warp_size = (image_shape[1] // 2 + shift_of_view * 2, image_shape[0] * 2)

    return mat, warp_size


class VisualizationRender(object):
    def __init__(self, frames_shape: Tuple[int, int],
                 pov: float = 90,
                 camera_h:
                 float = 2.0,
                 angle: float = 2.5,
                 dists: Optional[List[int]] = None,
                 train_asset_path: Optional[str] = None):
        self.dists = dists if dists is not None else [10, 20, 100]
        self.train_asset_path = train_asset_path if train_asset_path is not None else os.path.join(
            os.path.dirname(__file__), '../assets/train_view.png'
        )

        self.dist_info = {
            'K': get_intrinsics(pov, frames_shape[0], frames_shape[1]),
            'angle': transform_to_formula_angle(angle),
            'h': camera_h,
            'distances': self.dists,
            'stream_shape': frames_shape,
            'POV': pov
        }

        self.train_asset: Optional[np.ndarray] = None
        self.non_warped_lines_asset: Optional[np.ndarray] = None
        self.warped_lines_asset: Optional[np.ndarray] = None

        self.load_train_asset()
        self.warp_matrix, self.warp_size = gen_warp_matrix((frames_shape[0], frames_shape[1]))

    def load_train_asset(self):
        self.train_asset = cv2.imread(self.train_asset_path, cv2.IMREAD_UNCHANGED)
        self.train_asset = self.train_asset[:self.train_asset.shape[0] // 3]
        self.train_asset = cv2.resize(self.train_asset, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_AREA)

    def __call__(self, frame: np.ndarray, detections: List[Tuple[List[int], float, int]]) -> np.ndarray:
        """
        Draw line on input stream frame
        Args:
            frame: BGR OpenCV image
            detections: YOLO detections in (BOX, CONF, CLASS) format

        Returns:
            BGR OpenCV image
        """
        out, self.warped_lines_asset = render_dist_frame(
            frame, self.train_asset, self.warp_matrix, self.warp_size,
            detections, self.dist_info, self.warped_lines_asset
        )
        frame, self.non_warped_lines_asset = draw_lines(
            frame,
            self.dist_info['K'],
            self.dist_info['angle'],
            self.dist_info['distances'],
            self.dist_info['h'],
            swapRB=True, non_warped_lines=self.non_warped_lines_asset
        )
        out = merge_frames(frame, out)
        return out
