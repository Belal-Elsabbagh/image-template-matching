import numpy as np
from cv2.mat_wrapper import Mat

from src.image_ops import match_template, get_shape, rotate_image
from src.image_ops.scale_img import scale_img


def get_matches(img: Mat, template: Mat, threshold: float) -> list[tuple]:
    """Gets a list of matches for the template in the image.

    Args:
        img (Mat): The image to search for the template in.
        template (Mat): The template to search for in the image.
        threshold (float, optional): Minimum ratio of similarity. Defaults to 0.5.

    Returns:
        list[tuple]: Returns a list of the x and y coordinates of the top left corner of any found matches.
    """
    res = match_template(img, template)
    loc = np.where(res >= threshold)
    points = [*zip(*loc[::-1])]
    return [] if len(loc[0]) == len(loc[1]) == 0 else points


def get_matches_size_flexible(img: Mat, template: Mat, threshold: float):
    results = []
    img_shape, tmp_shape = get_shape(img), get_shape(template)
    scale = 0.9
    while img_shape.height > tmp_shape.height and img_shape.width > tmp_shape.width:
        results += get_matches(img, template, threshold)
        img = scale_img(img, scale)
        img_shape = get_shape(img)
    return results


def get_matches_rotation_flexible(img: Mat, template: Mat, threshold: float):
    results = []
    for angle in range(1, 360, 2):
        results += get_matches(img, template, threshold)
        img = rotate_image(img, angle)
    return results
