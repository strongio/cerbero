from typing import Dict, List, Tuple, Union, Set

import numpy as np
import pandas as pd
import torch
from torch import Tensor

TensorCollection = Union[Tensor, dict, list, tuple]


def list_to_tensor(item_list: List[Tensor]) -> Tensor:
    """Convert a list of Tensor into a single Tensor."""

    # Convert single value tensor
    if all(item_list[i].dim() == 0 for i in range(len(item_list))):
        item_tensor = torch.stack(item_list, dim=0)
    # Convert 2 or more-D tensor with the same shape
    elif all(
        (item_list[i].size() == item_list[0].size()) and (len(item_list[i].size()) != 1)
        for i in range(len(item_list))
    ):
        item_tensor = torch.stack(item_list, dim=0)
    # Convert reshape to 1-D tensor and then convert
    else:
        item_tensor, _ = pad_batch([item.view(-1) for item in item_list])

    return item_tensor


def pad_batch(
    batch: List[Tensor],
    max_len: int = 0,
    pad_value: int = 0,
    left_padded: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Convert the batch into a padded tensor and mask tensor.

    Parameters
    ----------
    batch
        The data for padding
    max_len
        Max length of sequence of padding
    pad_value
        The value to use for padding
    left_padded
        If True, pad on the left, otherwise on the right

    Returns
    -------
    Tuple[Tensor, Tensor]
        The padded matrix and correspoing mask matrix.
    """

    batch_size = len(batch)
    max_seq_len = int(np.max([len(item) for item in batch]))  # type: ignore

    if max_len > 0 and max_len < max_seq_len:
        max_seq_len = max_len

    padded_batch = batch[0].new_full((batch_size, max_seq_len), pad_value)

    for i, item in enumerate(batch):
        length = min(len(item), max_seq_len)  # type: ignore
        if left_padded:
            padded_batch[i, -length:] = item[-length:]
        else:
            padded_batch[i, :length] = item[:length]

    mask_batch = torch.eq(padded_batch.clone().detach(), pad_value).type_as(
        padded_batch
    )

    return padded_batch, mask_batch


def move_to_device(
    obj: TensorCollection, device: int = -1
) -> TensorCollection:  # pragma: no cover
    """Recursively move Tensors to a given CUDA device.

    Given a structure (possibly) containing Tensors on the CPU, move all the Tensors
    to the specified GPU (or do nothing, if they should beon the CPU).

    Originally from:
    https://github.com/HazyResearch/metal/blob/mmtl_clean/metal/utils.py

    Parameters
    ----------
    obj
        Tensor or collection of Tensors to move
    device
        Device to move Tensors to
        device = -1 -> "cpu"
        device =  0 -> "cuda:0"
    """

    if device < 0 or not torch.cuda.is_available():
        return obj
    elif isinstance(obj, Tensor):
        return obj.cuda(device)  # type: ignore
    elif isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(item, device) for item in obj]
    elif isinstance(obj, tuple):
        return tuple([move_to_device(item, device) for item in obj])
    else:
        return obj


def collect_flow_outputs_by_suffix(
    output_dict: Dict[str, Tensor], suffix: str
) -> List[Tensor]:
    """Return output_dict outputs specified by suffix, ordered by sorted flow_name."""
    return [
        output_dict[flow_name]
        for flow_name in sorted(output_dict.keys())
        if flow_name.endswith(suffix)
    ]


def metrics_dict_to_dataframe(metrics_dict: Dict[str, float]) -> pd.DataFrame:
    """Format a metrics_dict (with keys 'label/dataset/split/metric') format as a pandas DataFrame."""

    metrics = []

    for full_metric, score in metrics_dict.items():
        label_name, dataset_name, split, metric = tuple(full_metric.split("/"))
        metrics.append((label_name, dataset_name, split, metric, score))

    return pd.DataFrame(
        metrics, columns=["label", "dataset", "split", "metric", "score"]
    )


class Database:
    """A basic relational database that allows multiple keys for one value"""

    def __init__(self) -> None:
        self.keys: Dict[str, Tensor] = dict()
        self.values: Dict[Tensor, Set[str]] = dict()

    def __getitem__(self, key: str) -> Tensor:  # <---SQL SELECT statement
        return self.keys[key]

    def __setitem__(self, key: str, value: Tensor) -> None:
        if key not in self.keys:  # it's a new key <---SQL INSERT statement
            if value not in self.values:  # it's a new value
                self.keys[key] = value
                self.values[value] = set()  # a new set
                self.values[value].add(key)
            elif value in self.values:
                self.keys[key] = value  # a new key
                self.values[value].add(key)  # but just an update to the values
        elif key in self.keys:  # it's a new relationships
            self.keys[key] = value
            if value not in self.values:
                self.values[value] = set()
                self.values[value].add(key)
            elif value in self.values:
                self.values[value].add(key)

    def update(self, key: str, old_value: Tensor, new_value: Tensor) -> None:
        """update is a special case because __setitem__ can't see that
        you want to propagate your update onto multiple values. """
        if old_value == self.keys[key]:
            affected_keys = self.values[old_value]
            for key in affected_keys:
                self.__setitem__(key, new_value)
            del self.values[old_value]
        else:
            raise KeyError("key: {} does not have value: {}".format(key, old_value))

    def __delitem__(self, key: str) -> None:  # <---SQL DELETE statement
        try:
            value = self.keys[key]
            self.values[value].remove(key)
            if not self.values[value]:
                del self.values[value]
            del self.keys[key]  # then we delete the key.
        except KeyError:
            raise KeyError("key not found")

    def iterload(self, key_list: List[str], value_list: List[Tensor]) -> None:
        for key in key_list:
            for value in value_list:
                self.__setitem__(key, value)
