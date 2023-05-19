import numpy as np
from PIL import Image


def squeeze_with_alpha(image: Image.Image):
    mask = (np.array(image.convert('RGBA'))[:, :, 3].astype(float) / 255.0) > 0.5
    v_xs, = np.where(mask.sum(axis=0) > 0)
    v_ys, = np.where(mask.sum(axis=1) > 0)
    x0, x1 = v_xs.min(), v_xs.max()
    y0, y1 = v_ys.min(), v_ys.max()

    return image.crop((x0, y0, x1, y1))
