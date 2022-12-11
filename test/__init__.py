from timeit import default_timer

from src.image_ops import read_img, show_img
from src.template_matching import match_template_to_img_size_flexible, \
    match_template_to_img_rotation_flexible, match_template_to_img


def _test_template_matching(img_path, template_path, threshold=0.8, best_only=True, matching_function=None):
    if matching_function is None:
        matching_function = match_template_to_img
    test_img = read_img(img_path)
    test_template = read_img(template_path)
    return matching_function(test_img, test_template, threshold, best_only)


def test_template_matching(img_path, template_path, threshold=0.8, best_only=True):
    start = default_timer()
    result = _test_template_matching(img_path, template_path, threshold, best_only, match_template_to_img)
    print(f'standard template matching done in {round(default_timer() - start, 2)} seconds')
    show_img(f'found matches of {template_path} in {img_path}', result)


def test_size_flexible_template_matching(img_path, template_path, threshold=0.8, best_only=True):
    start = default_timer()
    result = _test_template_matching(img_path, template_path, threshold, best_only, match_template_to_img_size_flexible)
    print(f'size flexible matching done in {round(default_timer() - start, 2)} seconds')
    show_img(f'found matches of {template_path} in {img_path}', result)


def test_rotation_flexible_template_matching(img_path, template_path, threshold=0.8, best_only=True):
    start = default_timer()
    result = _test_template_matching(img_path, template_path, threshold, best_only,
                                     match_template_to_img_rotation_flexible)
    print(f'rotation flexible matching done in {round(default_timer() - start, 2)} seconds')
    show_img(f'found matches of {template_path} in {img_path}', result)
