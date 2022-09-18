# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch


class myDatasetConfig(object):
    def __init__(self):
        self.num_class = 21

        # Class name and id map
        # see: https://github.com/GeeeG/CloudPose/blob/d6410dc4af9a58c00511e34fdc41c6cfd9f96cba/ycb_video_data_tfRecords/script/2_dataset_to_tfRecord_small.py
        self.class_id = {
            "master chef can": 0,
            "cracker box": 1,
            "suger box": 2,
            "tomato soup can": 3,
            "mustard bottle": 4,
            "tuna fish can": 5,
            "pudding box": 6,
            "gelatin box": 7,
            "potted meat can": 8,
            "banana": 9,
            "pitcher base": 10,
            "bleach cleanser": 11,
            "bowl": 12,
            "mug": 13,
            "drill": 14,
            "wood block": 15,
            "scissors": 16,
            "large marker": 17,
            "large clapm": 18,
            "extra large clamp": 19,
            "foam brick": 20,
        }

        self.id_class = {self.class_id[t]: t for t in self.class_id}

        # 2D array
        self.onehot_encoding = torch.eye(self.num_class)[
            torch.tensor([range(self.num_class)]).reshape(-1)
        ]

    def sem2class(self, cls):
        # Select ith row of the 2D array
        onehot = self.onehot_encoding[int(cls), :]
        return onehot

    def size2class(self, class_name):
        """Convert 3D box size (l,w,h) to size class and size residual"""
        size_class = self.class_id[class_name]  # 0
        # size_residual = size - self.type_mean_size[type_name]  # 尺寸
        return size_class

    def class2size(self, pred_cls):
        """Inverse function to size2class"""
        mean_size = self.type_mean_size[self.id_class[pred_cls]]
        return mean_size

    def class2sem(self, pred_cls):
        """Given point_cloud_with_cls, return class name"""
        class_id = torch.argwhere(pred_cls[0, -self.num_class :] == 1).flatten()[0]
        return class_id

    def class2id(self, one_hot_cls):
        """Given one_hot_cls, return class id"""
        class_id = torch.argwhere(one_hot_cls == 1)[:, 1]
        return class_id

    def id2class(self, cls):
        """Return class name given class id."""
        return self.id_class[cls]
