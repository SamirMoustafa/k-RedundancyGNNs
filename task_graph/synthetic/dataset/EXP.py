import json
import os
from collections import defaultdict
from pathlib import Path

from numpy import arange, delete
from torch import float32, load, long, save, tensor
from torch_geometric.data import Data, InMemoryDataset

NAME = "GRAPHSAT"


def to_data(dic):
    tensor_data = {}
    for key, value in dic.items():
        tensor_data[key] = tensor(value, dtype=long if key in ["edge_index", "y"] else float32)
    return Data(**tensor_data)


class EXP(InMemoryDataset):
    def __init__(
        self,
        root=Path(__file__).resolve().parents[3].joinpath("data", "EXP"),
        task="iso",
        n_splits=4,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.root = root / ("EXP" if task == "iso" else "CEXP")
        super(EXP, self).__init__(self.root, transform, pre_transform, pre_filter)
        self.data, self.slices = load(self.processed_paths[0])
        self.n_splits = n_splits

    @property
    def raw_file_names(self):
        return [NAME + ".pkl"]

    @property
    def processed_file_names(self):
        return "data.pt"

    def download(self):
        pass

    def process(self):
        # Read dataset from a text (JSON) file
        with open(os.path.join(self.root, "raw/" + NAME + ".json")) as f:
            data_list = json.load(f)
            data_list = [to_data(dic) for dic in data_list]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        save((data, slices), self.processed_paths[0])

    def get_all_splits_idx(self):
        splits = defaultdict(list)
        val_size = int(self.__len__() * 0.1)
        test_size = int(self.__len__() * 0.15)
        for it in range(self.n_splits):
            indices = arange(self.__len__())
            val_idx = arange(start=it * val_size, stop=(it + 1) * val_size)
            test_idx = arange(start=it * test_size, stop=(it + 1) * test_size)
            splits["val"].append(indices[val_idx])
            remaining_indices = delete(indices, val_idx)
            splits["test"].append(remaining_indices[test_idx])
            remaining_indices = delete(remaining_indices, test_idx)
            splits["train"].append(remaining_indices)
        return splits


if __name__ == "__main__":
    dataset = EXP()
    print(dataset[0])
    dataset = EXP(task="class")
    print(dataset[0])
    splits = dataset.get_all_splits_idx()
    print(splits)
