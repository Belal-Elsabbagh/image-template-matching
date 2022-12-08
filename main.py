from test import test_template_matching

if __name__ == '__main__':
    rainbow_path = r'data\rainbow.png'
    rainbow_template_path = r'data\rainbow_template.png'
    color_pencils_path = r'data\color_pencils.jpeg'
    color_pencils_template_path = r'data\color_pencils_template.jpeg'

    test_template_matching(rainbow_path, rainbow_template_path)
    test_template_matching(color_pencils_path, color_pencils_template_path)
    test_template_matching(rainbow_path, color_pencils_template_path, 0.7, False)