import hydra
import ptflops
import torch

from models.CloudPose import CloudPose


@hydra.main(version_base=None, config_path="conf", config_name="config")
def compute_ptflops(cfg):
    with torch.cuda.device(0):
        net = CloudPose(
            backbone="pointcloudtransformer",
            cfg=cfg,
            channel=cfg.data.channel,
            num_class=cfg.data.num_class,
        )
        macs, params = ptflops.get_model_complexity_info(
            net, (1024, 24), as_strings=True, print_per_layer_stat=True, verbose=True
        )
        print("{:<30}  {:<8}".format("Computational complexity: ", macs))
        print("{:<30}  {:<8}".format("Number of parameters: ", params))


if __name__ == "__main__":
    compute_ptflops()
