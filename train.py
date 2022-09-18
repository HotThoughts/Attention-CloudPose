import json
import os
import sys
from typing import Dict, Optional

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch3d import transforms
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from tfrecord.torch.dataset import MultiTFRecordDataset

from models.CloudPose import CloudPose
from mydataset.model_util_my import myDatasetConfig
from utils import pc_util

# ---------------- GLOBAL CONFIG BEG
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if os.name == "posix":
    # data is mounted on datastores if using grid.ai
    ROOT_DIR = "/datastores/fps1024"
else:
    ROOT_DIR = BASE_DIR
    sys.path.append(os.path.join(ROOT_DIR, "mydataset"))
sys.path.append(os.path.join(ROOT_DIR, "utils"))
sys.path.append(os.path.join(ROOT_DIR, "models"))

DC = myDatasetConfig()  # Dataset specific Config
MEAN_COLOR_RGB = np.array([0.5, 0.5, 0.5])  # sunrgbd color is in 0~1


class FPS1024DataModule(pl.LightningDataModule):
    """Dataloader for FPS1024 dataset."""

    def __init__(self, batch_size=32, num_workers=4, tfrecord_dir="mydataset"):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tfrecord_dir = tfrecord_dir
        self.shuffle_queue_size = 1024
        self.description = {
            "xyz": "float",
            "rgb": "float",
            "translation": "float",
            "quaternion": "float",
            "num_valid_points_in_segment": "int",
            "seq_id": "int",
            "frame_id": "int",
            "class_id": "int",
        }
        SPLITS_PATH = os.path.join(ROOT_DIR, "FPS1024.json")
        self.splits = json.load(
            open(
                SPLITS_PATH,
                "r",
            )
        )

    def setup(self, stage: Optional[str] = None):
        self.train = self._multiTFRecordDataset("train")
        self.test = self._multiTFRecordDataset("test")
        self.val = self._multiTFRecordDataset("val")

    def _multiTFRecordDataset(self, dir):
        return MultiTFRecordDataset(
            data_pattern=os.path.join(self.tfrecord_dir, dir, "{}.tfrecords"),
            index_pattern=os.path.join(self.tfrecord_dir, dir, "{}.idx"),
            splits=self.splits,
            description=self.description,
            shuffle_queue_size=self.shuffle_queue_size,
            transform=self._transform_input,
            infinite=False,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def _attach_class(self, point_cloud, class_id):
        one_hot = DC.sem2class(class_id)
        one_hot_ex_rep = np.repeat(np.expand_dims(one_hot, axis=0), 1024, axis=0)
        point_cloud_with_cls = np.concatenate((point_cloud, one_hot_ex_rep), axis=1)
        return point_cloud_with_cls

    def deattach_class(self, point_cloud_with_cls):
        class_id = torch.argwhere(point_cloud_with_cls[0, -21:] == 1).flatten()[0]
        point_cloud = point_cloud_with_cls[:, :3]
        return point_cloud, class_id

    def _transform_input(self, x: Dict, use_color: bool = False) -> Dict:
        point_cloud = x["xyz"].reshape((1024, 3))
        point_cloud = pc_util.random_sampling(point_cloud, 1024)

        if not use_color:
            point_cloud = point_cloud[:, 0:3]
        else:
            point_cloud = point_cloud[:, 0:6]
            point_cloud[:, 3:] = point_cloud[:, 3:] - MEAN_COLOR_RGB
        point_cloud_with_cls = torch.from_numpy(
            self._attach_class(point_cloud, x["class_id"])
        )

        axag = transforms.quaternion_to_axis_angle(torch.from_numpy(x["quaternion"]))

        return {
            "xyz": torch.from_numpy(point_cloud),
            "point_clouds": point_cloud_with_cls.type(torch.float32),
            "axag_label": axag,
            "translate_label": torch.from_numpy(x["translation"]).type(torch.float32),
        }


def prepare_model_and_trainer(cfg: DictConfig):
    # Init model
    if cfg.train.baseline:
        model = CloudPose(
            backbone="pointnet",
            cfg=cfg,
            channel=cfg.data.channel,
            num_class=cfg.data.num_class,
        )
        wandb_tags = ["cloudpose", "baseline"]
    else:
        model = CloudPose(
            backbone="pointcloudtransformer",
            cfg=cfg,
            channel=cfg.data.channel,
            num_class=cfg.data.num_class,
        )
        wandb_tags = ["attention cloudpose", "transformer"]

    if cfg.train.fast_mode:
        LOG_EVERY_N_STEPS = 1
        os.environ["WANDB_MODE"] = "offline"
    elif cfg.test:
        LOG_EVERY_N_STEPS = 500
        os.environ["WANDB_MODE"] = "online"
    else:
        LOG_EVERY_N_STEPS = 500
        os.environ["WANDB_MODE"] = "online"

    assert len(cfg.train.lr_decay_steps) == len(cfg.train.lr_decay_rate)

    # Define Pytorch lightning trainer
    wandb_logger = WandbLogger(
        project=cfg.train.wandb.project,
        save_dir=cfg.train.wandb.save_dir,
        log_model=cfg.train.wandb.log_model,
        tags=wandb_tags,
    )
    trainer = pl.Trainer(
        auto_scale_batch_size="binsearch",
        auto_lr_find=False,
        default_root_dir=cfg.train.wandb.save_dir,
        devices=-1,  # use all available GPUs
        auto_select_gpus=True,
        accelerator="cuda",
        fast_dev_run=cfg.train.fast_mode,
        precision=cfg.train.precision,
        logger=wandb_logger,
        check_val_every_n_epoch=1,
        log_every_n_steps=LOG_EVERY_N_STEPS,
        max_epochs=cfg.train.max_epochs,
        enable_progress_bar=True,
        enable_checkpointing=True,
        # profiler="simple",
        benchmark=False,
        callbacks=[ModelCheckpoint(save_top_k=10, monitor="val/acc_best", mode="max")],
    )
    # Log gradients, parameter, histogram and model topology
    wandb_logger.watch(model, log="all")
    return model, trainer


def train(cfg: DictConfig):
    model, trainer = prepare_model_and_trainer(cfg)

    # Prepare data
    fps1024 = FPS1024DataModule(
        batch_size=cfg.dataloader.batch_size,
        num_workers=cfg.dataloader.num_workers,
        tfrecord_dir=cfg.data.tfrecord_dir,
    )
    # Tune
    if cfg.train.tune_model:
        trainer.tune(model, datamodule=fps1024)

    # Train
    if cfg.train.resume_checkpoint and not cfg.train.fast_mode:
        trainer.fit(model, datamodule=fps1024, ckpt_path=cfg.train.checkpoint_path)
    else:
        trainer.fit(model, datamodule=fps1024)


def test(cfg: DictConfig):
    model, trainer = prepare_model_and_trainer(cfg)

    # Prepare data
    fps1024 = FPS1024DataModule(
        batch_size=cfg.dataloader.batch_size,
        num_workers=cfg.dataloader.num_workers,
        tfrecord_dir=cfg.data.tfrecord_dir,
    )
    trainer.test(model, datamodule=fps1024, ckpt_path=cfg.train.checkpoint_path)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run(cfg: DictConfig):
    if cfg.test:
        test(cfg)
    else:
        train(cfg)


if __name__ == "__main__":
    run()
