import random
import warnings

import numpy as np
from PIL import Image

from .heat import ch_heatmap


def character_layout(image: Image.Image, layout_ratio: float, char_ratio: float, delta: float = 0.02,
                     max_tries: int = 20000):
    heatmap, musts = ch_heatmap(image)

    heatmap_sum = heatmap.sum()
    min_char_ratio = (heatmap * musts.astype(np.uint8).astype(float)).sum() / heatmap_sum
    if char_ratio < min_char_ratio:
        warnings.warn(f'Min character ratio is {min_char_ratio:.4f}, '
                      f'given value {char_ratio:.4f} will be replaced.')
        char_ratio = min_char_ratio

    x_xs, = np.where(musts.sum(axis=0) > 0)
    x_x0, x_x1 = x_xs.min(), x_xs.max()
    x_ys, = np.where(musts.sum(axis=1) > 0)
    x_y0, x_y1 = x_ys.min(), x_ys.max()

    p_size = int(((image.width * image.height) / layout_ratio * 2) ** 0.5)
    p_size = max(p_size, int(image.width / layout_ratio), int(image.height / layout_ratio))
    p_image = Image.fromarray(np.zeros((p_size, p_size, 4), dtype=np.uint8), mode='RGBA')
    left, top = (p_size - image.width) // 2, (p_size - image.height) // 2
    p_image.paste(image, (left, top, left + image.width, top + image.height), mask=image)

    p_heatmap = np.zeros((p_image.height, p_image.width), dtype=heatmap.dtype)
    p_heatmap[top:top + heatmap.shape[0], left:left + heatmap.shape[1]] = heatmap
    px_x0, px_y0 = x_x0 + left, x_y0 + top
    px_x1, px_y1 = x_x1 + left, x_y1 + top
    p_x0, p_y0 = left, top
    p_x1, p_y1 = image.width + left, image.height + top

    tries = 0
    while True:
        x0 = random.choice(range(0, px_x0))
        x1 = random.choice(range(px_x1, p_image.width))
        y0 = random.choice(range(0, px_y0))
        y1 = random.choice(range(px_y1, p_image.height))

        c_char_ratio = p_heatmap[y0:y1, x0:x1].sum() / heatmap_sum
        o_x0, o_x1 = max(p_x0, x0), min(p_x1, x1)
        o_y0, o_y1 = max(p_y0, y0), min(p_y1, y1)
        c_layout_ratio = ((o_x1 - o_x0) * (o_y1 - o_y0)) / ((x1 - x0) * (y1 - y0))

        if char_ratio - delta <= c_char_ratio <= char_ratio + delta and \
                layout_ratio - delta <= c_layout_ratio <= layout_ratio + delta:
            return p_image.crop((x0, y0, x1, y1))

        tries += 1
        if tries > max_tries:
            assert False, f'Max tried exceeded - {tries}/{max_tries}.'
