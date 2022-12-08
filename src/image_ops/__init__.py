import os

import cv2
import numpy as np
from cv2.mat_wrapper import Mat

from src.image_ops.get_shape import ImageShape


def get_shape(img):
    """
    Gets an object with data of the image's shape
    """
    return ImageShape(img)


def read_images_from_directory(dir_path, color_mode):
    return {f: read_img(os.path.join(dir_path, f), color_mode) for f in os.listdir(dir_path)}


def read_img(file_path, color_mode=None):
    if color_mode is None:
        color_mode = cv2.IMREAD_UNCHANGED
    return cv2.imread(file_path, color_mode)


def read_img_color(file_path):
    return read_img(file_path, cv2.IMREAD_COLOR)


def read_img_grayscale(file_path):
    return read_img(file_path, cv2.IMREAD_GRAYSCALE)


def show_img(window_name, img):
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return None


def save_image(file_path, img):
    return cv2.imwrite(file_path, img)


def blur_image(_img, size):
    return cv2.blur(_img, (size, size))


def filter_image(_img, _kernel):
    return cv2.filter2D(_img, -1, _kernel)


def match_template(img, template):
    return cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)


def draw_rectangle_from_detection(_img, res, color=(0, 255, 0), thickness=2):
    img = _img
    if res is None:
        return _img
    for x, y, w, h in res:
        draw_rectangle_on_image(img, (x,y), w, h)
    return img


def draw_rectangle_on_image(img, pt, w, h, color=(0, 255, 0), thickness=2):
    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), color, thickness)


def convert_to_grayscale(img) -> Mat:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def convert_to_float32(img):
    return np.float32(img)


def get_image_min_max_levels(img):
    return np.min(img), np.max(img)
