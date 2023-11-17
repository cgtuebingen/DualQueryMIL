from typing import Any, List, Type

import torch
from torch.nn import functional as F
from pytorch_lightning import LightningModule
from torchmetrics import MetricCollection, AUROC,  MaxMetric, MeanMetric, CohenKappa, F1Score, Recall, Precision, Specificity
from torchmetrics.classification.accuracy import Accuracy

class EnsembleVotingModel(LightningModule):
    def __init__(self, model_cls: Type[LightningModule], checkpoint_paths: List[str]) -> None:
        super().__init__()
        # Create `num_folds` models with their associated fold weights
        self.models = torch.nn.ModuleList([model_cls.load_from_checkpoint(p) for p in checkpoint_paths])
        self.ensemble_loss = MeanMetric()
        self.ensemble_metrics = self.models[0].metric.clone(prefix="ensemble/")
        # self.test_acc = Accuracy(task="multiclass", num_classes=10)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        # Compute the averaged predictions over the `num_folds` models.
        loss, logits, Y_prob, Y_hat, targets, attention = torch.stack([model.step(batch[0]) for model in self.models]).mean(0)
        self.ensemble_loss(loss)
        self.log("ensemble/loss", self.ensemble_loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": Y_prob, "targets": targets}
    
    def test_epoch_end(self, outputs: List[Any]):
        preds = torch.cat([output['preds'] for output in outputs], dim = 0).squeeze()
        targets = torch.stack([output['targets'] for output in outputs], dim = 0).squeeze()
        self.ensemble_metrics(preds , targets)
        self.log_dict(self.ensemble_metrics, on_step=False, on_epoch = True, prog_bar=True)
