import os
import sys
from types import List

import numpy as np
import torch
from tfrecord.torch.dataset import TFRecordDataset
from torch.utils.data import Dataset

from mydataset.model_util_my import myDatasetConfig

file = "../mydataset/FPS1024/train_files_FPS1024_0_0.tfrecords"
index_path = None
description = {
    "xyz": "float",
    "rgb": "float",
    "translation": "float",
    "quaternion": "float",
    "num_valid_points_in_segment": "int",
    "seq_id": "int",
    "frame_id": "int",
    "class_id": "int",
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, "utils"))

DC = myDatasetConfig()  # Dataset specific Config
MEAN_COLOR_RGB = np.array([0.5, 0.5, 0.5])  # sunrgbd color is in 0~1


class YCBTFRecordDataLoader(Dataset):
    def __init__(
        self,
        tfrecords_dir: str,
        description: dict,
        use_color: bool = False,
        scan_idx_list: List = None,
        batch_size: int = 1,
    ) -> None:
        super().__init__()
        self.data_path = os.path.join(BASE_DIR, tfrecords_dir)
        self.dataset = TFRecordDataset(
            self.data_path,
            index_path=None,
            description=description,
        )
        self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size)

    def __len__(self) -> int:
        """Iteratively count the number of items in the dataset."""
        return 1

    def __getitem__(self, index: int) -> dict:
        return next(iter(self.loader))
