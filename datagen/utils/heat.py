from typing import Tuple

import numpy as np
from PIL import Image
from imgutils.detect import detect_faces


def ch_heatmap(image: Image.Image, coef: float = 0.45, must_threshold: float = 0.85,
               r_max=1.0, r_min=0.15, p_min=0.3, p_max=4.0) -> Tuple[np.ndarray, np.ndarray]:
    alphas = np.array(image.convert('RGBA'))[:, :, 3].astype(float) / 255.0
    faces = detect_faces(image)
    assert len(faces) > 0, f'At least one face should be detected, but not found - {image!r}.'
    (x0, y0, x1, y1), _, _ = faces[0]

    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    f_width, f_height = x1 - x0, y1 - y0
    f_size = max(f_width, f_height)

    # y = kx + b
    # (x, y) = (0.5, r_max)
    # (x, y) = (p_range, r_min)
    k = (r_min - r_max) / (p_max - p_min)
    # b = y - kx
    b = r_min - p_max * k

    r_width = np.abs(np.arange(image.width) - cx) / f_size
    v_width = np.clip(r_width * k + b, a_min=r_min, a_max=r_max)

    r_height = np.abs(np.arange(image.height) - cy) / f_size
    v_height = np.clip(r_height * k + b, a_min=r_min, a_max=r_max)

    v_all = np.minimum(v_height[..., None], v_width[None, ...]) ** (coef / 0.5)
    v_all = v_all * alphas
    v_must = v_all >= must_threshold

    return v_all, v_must
