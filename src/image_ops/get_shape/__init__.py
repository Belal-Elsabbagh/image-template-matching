import numpy as np


class ImageShape:
    def __init__(self, img):
        shape = img.shape
        ch = shape[2] if len(shape) == 3 else 1
        h, w = shape[0], shape[1]
        px_count = h * w
        self.height = h
        self.width = w
        self.channel_count = ch
        self.pixel_count = px_count
        self.min_level, self.max_level = np.min(img), np.max(img)

    def __repr__(self):
        data = {
            'height': self.height,
            'width': self.width,
            'channel_count': self.channel_count,
            'pixel_count': self.pixel_count,
            'min_level': self.min_level,
            'max_level': self.max_level,
        }
        return f"image shape: {data}"
