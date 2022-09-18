import torch
from torchmetrics import Metric


class Accuracy(Metric):
    def __init__(self, threshold, full_state_update=False):
        super().__init__()
        self.threshold = threshold
        self.add_state("correct", default=torch.zeros([21]), dist_reduce_fx="sum")
        self.add_state("total", default=torch.zeros([21]), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape[0] == target.shape[0]
        indexs, counts = target.unique(return_counts=True, dim=0)
        self.total[indexs] += counts

        preds = preds <= self.threshold
        self.correct[indexs] += (
            torch.Tensor(
                [torch.sum(preds[(target == n).nonzero().flatten()]) for n in indexs]
            )
            .reshape(counts.shape)
            .to(preds.device)
        )

    def compute(self):
        return self.correct.float() / self.total
