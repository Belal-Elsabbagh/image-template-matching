from src.image_ops import read_img, show_img
from src.template_matching import match_template_to_img


def test_template_matching(img_path, template_path, threshold=0.8,best_only=True):
    test_img = read_img(img_path)
    test_template = read_img(template_path)
    result = match_template_to_img(test_img, test_template, threshold, best_only)
    if result is None:
        print(f'{template_path} does not exist in {img_path}')
        return None
    show_img(f'detected {template_path} in {img_path}', result)
