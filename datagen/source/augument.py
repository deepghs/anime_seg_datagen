import random
from typing import Tuple

import numpy as np
from PIL import Image
from imgutils.data import istack
from torchvision.transforms import ColorJitter as _OriginColorJitter
from torchvision.transforms import Compose

from ..utils import squeeze_with_alpha


class RandomRotate:
    def __init__(self, degrees: Tuple[float, float] = (-30, 30)):
        self._min_degree, self._max_degree = degrees

    def __call__(self, image: Image.Image) -> Image.Image:
        d = random.random() * (self._max_degree - self._min_degree) + self._min_degree
        return image.rotate(d, expand=True)


class ColorJitter:
    # keep alpha in images
    def __init__(self, brightness: float = 0, contrast: float = 0, saturation: float = 0, hue: float = 0):
        self.jitter = _OriginColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, image: Image.Image) -> Image.Image:
        arr = np.array(image.convert('RGBA'))
        alphas = arr[:, :, 3].astype(float) / 255.0
        rgb_image = Image.fromarray(arr[:, :, :3], mode='RGB')
        rgb_image = self.jitter(rgb_image)
        return istack((rgb_image, alphas))


DEFAULT_CH_AUG = Compose([
    RandomRotate((-45, 45)),
    ColorJitter(0.1, 0.1, 0.2, 0.2),
    squeeze_with_alpha,
])

DEFAULT_BG_AUG = Compose([
    ColorJitter(0.1, 0.1, 0.1, 0.1),
])
