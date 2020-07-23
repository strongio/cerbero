from typing import Any, Tuple

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

from tqdm.auto import tqdm

DataSet = Tuple[
    Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray],
]
DataSets = Tuple[DataSet, DataSet]


def make_cifar_dataset(train: bool = True, **kwargs: Any) -> DataSets:
    """Generate CIFAR10 dataset with RGB labels"""
    dataset = torchvision.datasets.CIFAR10(
        root="./data", train=train, download=True, transform=transforms.ToTensor()
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=2
    )

    norm_transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    images, class_labels, rgb_labels = [], [], []

    count = 0
    pbar = tqdm(total=len(dataloader))
    dataiter = iter(dataloader)
    while count < len(dataloader):
        _image, label = dataiter.next()

        # Compute RGB mean
        rgb_mean = _image[0].mean(dim=(1, 2))

        # Normalize image and add labels to list
        _image = norm_transform(_image[0])
        images.append(_image.numpy())
        class_labels.append(label.item())
        rgb_labels.append(rgb_mean.numpy())

        count += 1
        pbar.update(1)
    images = np.stack(images)
    class_labels = np.stack(class_labels)
    rgb_labels = np.stack(rgb_labels)

    return split_data(images, class_labels, rgb_labels, **kwargs)


def split_data(
    images: np.ndarray,
    class_labels: np.ndarray,
    rgb_labels: np.ndarray,
    seed: int = 123,
) -> DataSets:
    """Split dataset in half and return only partial labels for each"""
    images_1, images_2, class_labels_1, _, _, rgb_labels_2 = train_test_split(
        images, class_labels, rgb_labels, test_size=0.5, random_state=seed
    )
    return (images_1, class_labels_1), (images_2, rgb_labels_2)
