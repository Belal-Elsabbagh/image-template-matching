""" 
This module contains the main function for template matching.
"""
from cv2.mat_wrapper import Mat

from src.image_ops import get_shape, draw_rectangle_on_image, convert_to_grayscale
from src.template_matching.get_matches import get_matches, get_matches_size_flexible, get_matches_rotation_flexible


def match_template_to_img(img: Mat,
                          template: Mat,
                          threshold: float = 0.5,
                          best_only: bool = True, matching_method: callable=None) -> Mat:
    """Takes an image and a template and draws a rectangle around all matches of the template in the image above a
    certain similarity ratio.

    Args:
        img (Mat): The image to search for the template in.
        template (Mat): The template to search for in the image.
        threshold (float, optional): Minimum ratio of similarity. Defaults to 0.5.
        best_only (bool, optional): If True, returns only the best match if exists. Defaults to True.
        matching_method (callable, optional): The method used to get matches. Defaults to _get_matches.

    Returns:
        Mat: The image wih the detections drawn on it
    """
    img_gray, template = _init_images(img, template)
    if matching_method is None:
        matching_method = get_matches
    matches = matching_method(img_gray, template, threshold)
    if len(matches) == 0:
        return img
    if best_only:
        matches = [matches[0]]
    return _draw_detected_matches(img, template, matches)


def match_template_to_img_size_flexible(img: Mat,
                          template: Mat,
                          threshold: float = 0.5,
                          best_only: bool = True,):
    return match_template_to_img(img, template, threshold, best_only, get_matches_size_flexible)


def match_template_to_img_rotation_flexible(img: Mat,
                          template: Mat,
                          threshold: float = 0.5,
                          best_only: bool = True,):
    return match_template_to_img(img, template, threshold, best_only, get_matches_rotation_flexible)


def _draw_detected_matches(img: Mat, template: Mat, matches: list[tuple]) -> Mat:
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
