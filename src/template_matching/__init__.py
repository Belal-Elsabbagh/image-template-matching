""" 
This module contains the main function for template matching.
"""
import numpy as np
from cv2.mat_wrapper import Mat

from src.image_ops import get_shape, match_template, draw_rectangle_on_image, convert_to_grayscale


def match_template_to_img(img: Mat,
                          template: Mat,
                          threshold: float = 0.5,
                          best_only: bool = True) -> Mat:
    """Searches for the template in the image and draws a rectangle around it if found.

    Args:
        img (Mat): The image to search for the template in.
        template (Mat): The template to search for in the image.
        threshold (float, optional): Minimum ratio of similarity. Defaults to 0.5.
        best_only (bool, optional): If True, returns only the best match if exists. Defaults to True.

    Returns:
        Mat: The image with any found detections drawn on it.
    """
    img_gray, template = _init_images(img, template)
    res = _get_matches(img_gray, template, threshold, best_only)
    if len(res) == 0:
        return img
    return draw_detected_matches(img, template, res)


def draw_detected_matches(img: Mat, template: Mat, matches: list[tuple]) -> Mat:
    """Draws a rectangle around the given matches.

    Args:
        img (Mat): The main image.
        template (Mat): The template used in searching.
        matches (list[tuple]): A list of the x and y coordinates of the top left corner of any found matches.

    Returns:
        Mat: The image with the matches drawn on it.
    """
    shape = get_shape(template)
    for pt in matches:
        draw_rectangle_on_image(img, pt, shape.width, shape.height)
    return img


def _init_images(img: Mat, template: Mat) -> tuple[Mat,Mat]:
    """Converts the images to grayscale.

    Args:
        img (Mat): The original image.
        template (Mat): The template image.

    Returns:
        tuple[Mat,Mat]: A tuple of the grayscale image and the grayscale template.
    """
    return convert_to_grayscale(img), convert_to_grayscale(template)


def _get_matches(img: Mat, template: Mat, threshold: float, best_only: bool) -> list[tuple]:
    """Gets a list of matches for the template in the image.

    Args:
        img (Mat): The image to search for the template in.
        template (Mat): The template to search for in the image.
        threshold (float, optional): Minimum ratio of similarity. Defaults to 0.5.
        best_only (bool, optional): If True, returns only the best match if exists. Defaults to True.

    Returns:
        list[tuple]: Returns a list of the x and y coordinates of the top left corner of any found matches.
    """
    res = match_template(img, template)
    loc = np.where(res >= threshold)
    if len(loc[0]) == len(loc[1]) == 0:
        return []
    points = [*zip(*loc[::-1])]
    return [points[0]] if best_only else points
