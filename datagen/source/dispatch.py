import numpy as np
from torch.utils.data import Dataset

from .augument import DEFAULT_CH_AUG, DEFAULT_BG_AUG
from .skytnt import ImgMaskFgDataset, FgDataset, BgDataset


class _DatasetCombine(Dataset):
    def __init__(self, *ds: Dataset):
        self.ds = ds
        self._sizes = np.array([len(item) for item in self.ds], dtype=np.int32)

    def __len__(self):
        return self._sizes.sum()

    def _postprocess(self, index, image):
        return image

    def __getitem__(self, index):
        if index < 0:
            index = self._sizes.sum() + index

        current = 0
        for item, size in zip(self.ds, self._sizes):
            if current + size >= index:
                return self._postprocess(index, item[int(index - current)])

            current += size
        else:
            raise IndexError(f'{self._sizes.sum()!r} item(s) in dataset {self!r}, but {index!r} is given.')


class CharacterDataset(_DatasetCombine):
    def __init__(self, transform=DEFAULT_CH_AUG):
        _DatasetCombine.__init__(
            self,
            FgDataset(),
            ImgMaskFgDataset(),
        )
        self.transform = transform

    def _postprocess(self, index, image):
        if self.transform is not None:
            image = self.transform(image)
        return image


class BackgroundDataset(_DatasetCombine):
    def __init__(self, transform=DEFAULT_BG_AUG):
        _DatasetCombine.__init__(
            self,
            BgDataset(),
        )
        self.transform = transform

    def _postprocess(self, index, image):
        if self.transform is not None:
            image = self.transform(image)
        return image
