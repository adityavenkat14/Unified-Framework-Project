"""
my_datasets/imagenet100.py — ImageFolder-style dataset for the materialized
ImageNet-100 subset (see materialize_imagenet100.py). Matches the same
interface CUBDataset / OxfordIIITPet use (.classes, .class_to_idx,
__getitem__ -> (image, label)) so it drops into helper.py's load_dataset()
the same way.
"""

import json
import os
import pathlib

from torchvision.datasets import VisionDataset
from torchvision import datasets


class ImageNet100Dataset(VisionDataset):
    """
    root: /content/data/imagenet100 (contains classes.json, train/, validation/)
    split: "train" or "test" (mapped to "validation" internally -- see
           materialize_imagenet100.py's NOTE on why "test" isn't used directly)
    """

    def __init__(
        self,
        root: str,
        split: str = "test",
        transform=None,
        loader=datasets.folder.default_loader,
    ):
        super().__init__(root, transform=transform)
        self.loader = loader
        self._base_folder = pathlib.Path(root)

        with open(self._base_folder / "classes.json") as f:
            self.classes = json.load(f)  # already in correct label-index order
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}

        folder_name = "train" if split == "train" else "validation"
        split_dir = self._base_folder / folder_name

        if not split_dir.exists():
            raise RuntimeError(
                f"Split directory not found: {split_dir}. "
                f"Run materialize_imagenet100.py first."
            )

        self._images = []
        self._labels = []
        for class_name in self.classes:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue  # shouldn't happen if materialization completed cleanly
            label = self.class_to_idx[class_name]
            for img_path in sorted(class_dir.iterdir()):
                self._images.append(img_path)
                self._labels.append(label)

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        image = self.loader(self._images[idx])
        target = self._labels[idx]
        if self.transforms:
            image, target = self.transforms(image, target)
        return image, target
