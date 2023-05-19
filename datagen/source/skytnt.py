import os
from functools import lru_cache

import datasets
import numpy as np
from PIL import Image
from datasets import DownloadManager, DatasetInfo
from imgutils.data import istack
from torch.utils.data import Dataset

_DESCRIPTION = """\
A segmentation dataset for anime character
"""
_HOMEPAGE = "https://huggingface.co/datasets/skytnt/anime-segmentation"
_URL_BASE = "https://huggingface.co/datasets/skytnt/anime-segmentation/resolve/main/data/"
_EXTENSION = [".png", ".jpg"]


class AnimeSegmentationConfig(datasets.BuilderConfig):

    def __init__(self, features, data_files, **kwargs):
        super(AnimeSegmentationConfig, self).__init__(**kwargs)
        self.features = features
        self.data_files = data_files


class AnimeSegmentation(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        AnimeSegmentationConfig(
            name="bg",
            description="background",
            features=["image"],
            data_files=["bg-00.zip", "bg-01.zip", "bg-02.zip", "bg-03.zip", "bg-04.zip"]
        ),
        AnimeSegmentationConfig(
            name="fg",
            description="foreground",
            features=["image"],
            data_files=["fg-00.zip", "fg-01.zip", "fg-02.zip", "fg-03.zip", "fg-04.zip", "fg-05.zip"]
        ),
        AnimeSegmentationConfig(
            name="imgs-masks",
            description="real images and masks",
            features=["image", "mask"],
            data_files=["imgs-masks.zip"]
        )
    ]

    def _info(self) -> DatasetInfo:

        features = {feature: datasets.Image() for feature in self.config.features}
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(features),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation="",
        )

    def _split_generators(self, dl_manager: DownloadManager):
        urls = [_URL_BASE + data_file for data_file in self.config.data_files]
        dirs = dl_manager.download_and_extract(urls)
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"dirs": dirs})]

    def _generate_examples(self, dirs):
        if self.config.name != "imgs-masks":
            for path in dirs:
                all_fnames = {os.path.relpath(os.path.join(root, fname), start=path)
                              for root, _dirs, files in os.walk(path) for fname in files}
                image_fnames = sorted(fname for fname in all_fnames if os.path.splitext(fname)[1].lower() in _EXTENSION)
                for image_fname in image_fnames:
                    yield image_fname, {"image": os.path.join(path, image_fname)}
        else:
            path = dirs[0]
            all_fnames = {os.path.relpath(os.path.join(root, fname), start=path)
                          for root, _dirs, files in os.walk(os.path.join(path, "imgs")) for fname in files}
            image_fnames = sorted(fname for fname in all_fnames if os.path.splitext(fname)[1].lower() in _EXTENSION)
            for image_fname in image_fnames:
                yield image_fname, {"image": os.path.join(path, image_fname),
                                    "mask": os.path.join(path, image_fname.replace("imgs", "masks"))}


@lru_cache()
def _builder(name: str):
    return AnimeSegmentation(config_name=name)


@lru_cache()
def _init_builder(name: str):
    return _builder(name).download_and_prepare()


_NAMES = ['imgs-task', 'fg', 'bg']


@lru_cache()
def _dataset(name: str) -> Dataset:
    _init_builder(name)
    return _builder(name).as_dataset()['train']


class FgDataset(Dataset):
    def __init__(self):
        self.ds = _dataset('fg')

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, item) -> Image.Image:
        return self.ds[item]['image']


class ImgMaskFgDataset(Dataset):
    def __init__(self):
        self.ds = _dataset('imgs-masks')

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, item) -> Image.Image:
        data = self.ds[item]
        image, mask = data['image'], data['mask']
        image = image.convert('RGBA')
        mask = np.array(mask.convert('1')).astype(bool).astype(np.uint8).astype(float)
        return istack((image, mask))


class BgDataset(Dataset):
    def __init__(self):
        self.ds = _dataset('bg')

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, item) -> Image.Image:
        return self.ds[item]['image']
