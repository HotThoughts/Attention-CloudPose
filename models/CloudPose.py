import pytorch_lightning as pl
import torch
import torch.utils.data
from torchmetrics import MaxMetric

from models.PointCloudTransformer import PointCloudTransformer
from models.PointNet import PointNet
from mydataset.model_util_my import myDatasetConfig
from utils.evaluate import get_ADD_ADS
from utils.losses import get_loss
from utils.metric import Accuracy

DC = myDatasetConfig()  # Dataset specific Config


class CloudPose(pl.LightningModule):
    def __init__(self, cfg, backbone="pointnet", channel=3, num_class=21):
        super().__init__()
        self.num_class = num_class
        self.channel = channel
        self.cfg = cfg

        if backbone == "pointnet":
            self.trans = PointNet(self.channel, self.num_class)
            self.rot = PointNet(self.channel, self.num_class)
        elif backbone == "pointcloudtransformer":
            self.trans = PointCloudTransformer(self.channel, self.num_class)
            self.rot = PointCloudTransformer(self.channel, self.num_class)
        else:
            raise NotImplementedError(
                "The given backtone is undefined. Plese use 'pointnet' or 'pointcloudtransformer'"
            )
        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc_ad = Accuracy(threshold=cfg.accuracy.threshold)
        self.train_acc_ads = Accuracy(threshold=cfg.accuracy.threshold)
        # val
        self.val_acc_ad = Accuracy(threshold=cfg.accuracy.threshold)
        self.val_acc_ads = Accuracy(threshold=cfg.accuracy.threshold)
        # test
        self.test_acc_ad = Accuracy(threshold=cfg.accuracy.threshold)
        self.test_acc_ads = Accuracy(threshold=cfg.accuracy.threshold)
        self.test_acc_small_ad = Accuracy(threshold=cfg.accuracy.threshold_small)
        self.test_acc_small_ads = Accuracy(threshold=cfg.accuracy.threshold_small)

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()
        self.val_acc_ad_best = MaxMetric()
        self.val_acc_ads_best = MaxMetric()

        # save hyper-parameters to self.hparams (auto-logged by W&B)
        self.save_hyperparameters()

    def forward(self, point_clouds):
        """Define the prediction/inference actions."""
        point_clouds_tp = point_clouds.transpose(1, 2)  # b 8 256

        base_xyz = torch.mean(point_clouds_tp[:, : self.channel, :], dim=2)
        point_clouds_res = point_clouds_tp[:, : self.channel, :] - base_xyz.unsqueeze(
            -1
        )  # b 3 1
        point_clouds_res_with_cls = torch.cat(
            (point_clouds_res, point_clouds_tp[:, self.channel :, :]), dim=1
        )  # channel + cls

        t = self.trans(point_clouds_res_with_cls)
        r = self.rot(point_clouds_res_with_cls)

        return {"translate_pred": t + base_xyz, "axag_pred": r}

    def step(self, batch):
        x = batch["point_clouds"]
        end_points = self(x)
        for key in batch:
            assert key not in end_points
            end_points[key] = batch[key]
        loss, end_points = get_loss(end_points)
        point_class = end_points["point_clouds"][:, 0, 3:]
        return loss, end_points, point_class

    def configure_optimizers(self):
        if self.cfg.train.optimizer in ["Adam", "adam"]:
            optimizer = torch.optim.Adam(
                params=self.parameters(),
                lr=self.cfg.train.lr,
                weight_decay=self.cfg.train.weight_decay,
            )
        return optimizer

    def on_train_epoch_start(self) -> None:
        # Count training samples per class
        self.train_count_samples = torch.zeros([21]).cuda()
        return super().on_train_epoch_start()

    def training_step(self, batch, batch_idx):
        loss, end_points, point_class = self.step(batch)
        target = DC.class2id(point_class)
        target_cls, counts = target.unique(return_counts=True, dim=0)
        self.train_count_samples[target_cls] += counts
        prefix = "train"

        ad_loss, ads_loss = get_ADD_ADS(end_points, point_class)
        ad_acc = self.train_acc_ad(
            preds=ad_loss["per"],
            target=target,
        )
        ads_acc = self.train_acc_ads(
            preds=ads_loss["per"],
            target=target,
        )
        target_count = target_cls.size(dim=0)
        ad_acc_total = ad_acc.nansum(dim=0) / target_count
        ads_acc_total = ads_acc.nansum(dim=0) / target_count
        acc = (ad_acc_total + ads_acc_total) / 2

        self.log(f"{prefix}/acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            f"{prefix}/acc/AD",
            ad_acc_total,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            f"{prefix}/acc/ADS",
            ads_acc_total,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            f"{prefix}/loss/AD",
            ad_loss["total"],
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            f"{prefix}/loss/ADS",
            ads_loss["total"],
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            f"{prefix}/loss/translation",
            end_points["trans_loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            f"{prefix}/loss/rotation",
            end_points["axag_loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )

        for i in target_cls.squeeze().tolist():
            class_name = DC.id2class(i)
            if self.current_epoch >= 30:
                self.log(
                    f"{prefix}/acc/AD/{class_name}",
                    ad_acc[i],
                    on_epoch=True,
                    on_step=False,
                    prog_bar=False,
                )
                self.log(
                    f"{prefix}/acc/ADS/{class_name}",
                    ads_acc[i],
                    on_epoch=True,
                    on_step=False,
                    prog_bar=False,
                )
                self.log(
                    f"{prefix}/loss/AD/{class_name}",
                    ad_loss["cls"][i],
                    on_epoch=True,
                    on_step=False,
                    prog_bar=False,
                )
                self.log(
                    f"{prefix}/loss/ADS/{class_name}",
                    ads_loss["cls"][i],
                    on_epoch=True,
                    on_step=False,
                    prog_bar=False,
                )
            self.log(
                f"{prefix}/acc/AD/{class_name}",
                ad_acc[i],
                on_epoch=True,
                on_step=True,
                prog_bar=False,
            )
            self.log(
                f"{prefix}/acc/ADS/{class_name}",
                ads_acc[i],
                on_epoch=True,
                on_step=True,
                prog_bar=False,
            )
            self.log(
                f"{prefix}/loss/AD/{class_name}",
                ad_loss["cls"][i],
                on_epoch=True,
                on_step=True,
                prog_bar=False,
            )
            self.log(
                f"{prefix}/loss/ADS/{class_name}",
                ads_loss["cls"][i],
                on_epoch=True,
                on_step=True,
                prog_bar=False,
            )
            self.log(
                f"{prefix}/acc/AD/{class_name}",
                ad_acc[i],
                on_epoch=True,
                on_step=True,
                prog_bar=False,
            )
            self.log(
                f"{prefix}/acc/ADS/{class_name}",
                ads_acc[i],
                on_epoch=True,
                on_step=True,
                prog_bar=False,
            )
            self.log(
                f"{prefix}/loss/AD/{class_name}",
                ad_loss["cls"][i],
                on_epoch=True,
                on_step=True,
                prog_bar=False,
            )
            self.log(
                f"{prefix}/loss/ADS/{class_name}",
                ads_loss["cls"][i],
                on_epoch=True,
                on_step=True,
                prog_bar=False,
            )
            self.log(
                f"{prefix}/loss/translation/{class_name}",
                end_points["trans_clsLoss"][i],
                on_epoch=True,
                on_step=True,
                prog_bar=False,
            )
            self.log(
                f"{prefix}/loss/rotation/{class_name}",
                end_points["axag_clsLoss"][i],
                on_epoch=True,
                on_step=True,
                prog_bar=False,
            )
        return loss

    def training_epoch_end(self, outputs):
        for i in range(0, self.num_class):
            class_name = DC.id2class(i)
            self.log(f"train/count-samples/{class_name}", self.train_count_samples[i])
        self.train_acc_ad.reset()
        self.train_acc_ads.reset()

    def on_validation_epoch_start(self) -> None:
        self.val_count_samples = torch.zeros([21]).cuda()
        return super().on_validation_epoch_start()

    def validation_step(self, batch, batch_idx):
        loss, end_points, point_class = self.step(batch)
        ad_loss, ads_loss = get_ADD_ADS(end_points, point_class)
        target = DC.class2id(point_class)
        ad_acc = self.val_acc_ad(
            preds=ad_loss["per"],
            target=target,
        )
        ads_acc = self.val_acc_ads(
            preds=ads_loss["per"],
            target=target,
        )
        target_cls, counts = target.unique(return_counts=True, dim=0)
        self.val_count_samples[target_cls] += counts
        target_count = target_cls.size(dim=0)
        target_count = target_cls.size(dim=0)
        ad_acc_total = ad_acc.nansum(dim=0) / target_count
        ads_acc_total = ads_acc.nansum(dim=0) / target_count
        acc = (ad_acc_total + ads_acc_total) / 2

        prefix = "val"

        self.log(f"{prefix}/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            f"{prefix}/acc/AD",
            ad_acc_total,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            f"{prefix}/acc/ADS",
            ads_acc_total,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        for i in target_cls.squeeze().tolist():
            class_name = DC.id2class(i)
            self.log(
                f"{prefix}/acc/AD/{class_name}",
                ad_acc[i],
                on_epoch=True,
                on_step=False,
                prog_bar=False,
            )
            self.log(
                f"{prefix}/acc/ADS/{class_name}",
                ads_acc[i],
                on_epoch=True,
                on_step=False,
                prog_bar=False,
            )
        return loss

    def validation_epoch_end(self, outputs):
        acc_ad = self.val_acc_ad.compute()  # get val accuracy from current epoch
        acc_ads = self.val_acc_ads.compute()  # get val accuracy from current epoch
        acc_ad_total = acc_ad.mean(dim=0)
        acc_ads_total = acc_ads.mean(dim=0)
        acc = (acc_ad_total + acc_ads_total) / 2
        self.val_acc_ad_best.update(acc_ad_total)
        self.val_acc_ads_best.update(acc_ads_total)
        self.val_acc_best.update(acc)
        self.log(
            "val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True
        )
        self.log(
            "val/acc_ad_best",
            self.val_acc_ad_best.compute(),
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/acc_ads_best",
            self.val_acc_ads_best.compute(),
            on_epoch=True,
            prog_bar=True,
        )
        for i in range(0, self.num_class):
            class_name = DC.id2class(i)
            self.log(f"val/count-samples/{class_name}", self.val_count_samples[i])

        self.val_acc_ad.reset()
        self.val_acc_ads.reset()

    def on_test_epoch_start(self) -> None:
        self.test_count_samples = torch.zeros([21]).cuda()
        return super().on_test_epoch_start()

    def test_step(self, batch, batch_idx):
        loss, end_points, point_class = self.step(batch)
        ad_loss, ads_loss = get_ADD_ADS(end_points, point_class)
        target = DC.class2id(point_class)
        ad_acc = self.test_acc_ad(
            preds=ad_loss["per"],
            target=target,
        )
        ads_acc = self.test_acc_ads(
            preds=ads_loss["per"],
            target=target,
        )
        ad_acc_small = self.test_acc_small_ad(
            preds=ad_loss["per"],
            target=target,
        )
        ads_acc_small = self.test_acc_small_ads(
            preds=ads_loss["per"],
            target=target,
        )

        target_cls, counts = target.unique(return_counts=True, dim=0)
        self.test_count_samples[target_cls] += counts
        target_count = target_cls.size(dim=0)
        ad_acc_total = ad_acc.nansum(dim=0) / target_count
        ads_acc_total = ads_acc.nansum(dim=0) / target_count
        acc = (ad_acc_total + ads_acc_total) / 2

        ad_acc_small_total = ad_acc_small.nansum(dim=0) / target_count
        ads_acc_small_total = ads_acc.nansum(dim=0) / target_count
        acc_small = (ad_acc_small_total + ads_acc_small_total) / 2

        prefix = "test"

        self.log(f"{prefix}/acc", acc, on_epoch=True, on_step=True, prog_bar=True)
        self.log(
            f"{prefix}/acc_small",
            acc_small,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
        )
        self.log(f"{prefix}/loss/AD", ad_loss["total"], on_epoch=True, on_step=True)
        self.log(f"{prefix}/loss/ADS", ads_loss["total"], on_epoch=True, on_step=True)
        self.log(f"{prefix}/acc/AD", ad_acc_total, on_epoch=True, on_step=True)
        self.log(f"{prefix}/acc/ADS", ads_acc_total, on_epoch=True, on_step=True)
        self.log(
            f"{prefix}/acc/AD_small",
            ad_acc_small_total,
            on_epoch=True,
            on_step=True,
        )
        self.log(
            f"{prefix}/acc/ADS_small",
            ads_acc_small_total,
            on_epoch=True,
            on_step=True,
        )

        for i in target_cls.squeeze().tolist():
            class_name = DC.id2class(i)
            self.log(
                f"{prefix}/loss/AD/{class_name}",
                ad_loss["cls"][i],
                on_epoch=True,
                on_step=True,
            )
            self.log(
                f"{prefix}/loss/ADS/{class_name}",
                ads_loss["cls"][i],
                on_epoch=True,
                on_step=True,
            )
            self.log(
                f"{prefix}/acc/AD/{class_name}",
                ad_acc[i],
                on_epoch=True,
                on_step=True,
            )
            self.log(
                f"{prefix}/acc/ADS/{class_name}",
                ads_acc[i],
                on_epoch=True,
                on_step=True,
            )
            self.log(
                f"{prefix}/acc/AD_small/{class_name}",
                ad_acc_small[i],
                on_epoch=True,
                on_step=True,
            )
            self.log(
                f"{prefix}/acc/ADS_small/{class_name}",
                ads_acc_small[i],
                on_epoch=True,
                on_step=True,
            )
        return loss

    def test_epoch_end(self, outputs):
        for i in range(0, self.num_class):
            class_name = DC.id2class(i)
            self.log(f"test/count-samples/{class_name}", self.test_count_samples[i])
        self.test_acc_ad.reset()
        self.test_acc_ads.reset()
        self.test_acc_small_ad.reset()
        self.test_acc_small_ads.reset()
