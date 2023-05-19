import random

import numpy as np
from PIL import Image


def ch_image_with_background(ch_image: Image.Image, bg_image: Image.Image, ratio: float = 0.7):
    ch_image = ch_image.convert('RGBA')
    max_width, max_height = int(bg_image.width * ratio), int(bg_image.height * ratio)
    zoom = min(max_width / ch_image.width, max_height / ch_image.height)

    if zoom < 1.0:
        ch_image = ch_image.resize((int(ch_image.width * zoom), int(ch_image.height * zoom)))
    else:
        bg_image = bg_image.resize((int(bg_image.width / zoom), int(bg_image.height / zoom)))

    left = random.choice(range(bg_image.width - ch_image.width))
    top = random.choice(range(bg_image.height - ch_image.height))

    ret_image = bg_image.crop((left, top, left + ch_image.width, top + ch_image.height)).convert('RGBA')
    ret_image.paste(ch_image, (0, 0, ch_image.width, ch_image.height), mask=ch_image)

    ch_mask = (np.array(ch_image)[:, :, 3].astype(float) / 255.0) > 0
    return ret_image, ch_mask
