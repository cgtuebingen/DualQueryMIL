from typing import Any, List

from omegaconf.dictconfig import DictConfig
import torch
import torch.nn.functional as F
import pandas as pd
from pytorch_lightning import LightningModule
from torchmetrics import MetricCollection, AUROC,  MaxMetric, MeanMetric, CohenKappa, F1Score, Recall, Precision, Specificity, ConfusionMatrix
from torchmetrics.classification.accuracy import Accuracy

from src.optimizer.optim_factory import create_optimizer
from src.utils import plot_confmat


""" Cross-Entropy loss from 
https://github.com/ArchipLab-LinfengZhang/pytorch-self-distillation-final/blob/47ca93b6a60baaaf127a762e9f19bc70f4f0ffed/train.py#L28"""
def CrossEntropy(student_logits, teacher_logits):
    log_softmax_outputs = F.log_softmax(student_logits/3.0, dim=1)
    softmax_targets = F.softmax(teacher_logits/3.0, dim=1)
    return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()

class MilModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        name: str,
        slide_loss_fn: torch.nn.CrossEntropyLoss,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        illustrator: None
    ):
        """LightningModule for Multiple Instance Learning.

        Args:
            net (torch.nn.Module): _description_
            optimizer (torch.optim.Optimizer): _description_
            scheduler (torch.optim.lr_scheduler): _description_
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net

        # loss function
        self.slide_loss_fn = self.hparams.slide_loss_fn
        task = "multiclass"
        # metric objects for calculating and averaging accuracy across batches
        self.metric = MetricCollection([
            AUROC(task=task, num_classes=self.net.n_classes, average = 'macro'),
            Accuracy(task=task, num_classes=self.net.n_classes, average = 'micro'),
            F1Score(task=task, num_classes=self.net.n_classes, average = 'macro')])
        
        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_metrics = self.metric.clone(prefix="train/")
        self.val_metrics = self.metric.clone(prefix="val/")
        self.test_metrics = self.metric.clone(prefix="test/")

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far
        self.val_best_auroc = MaxMetric()
        self.val_best_acc = MaxMetric()

        # for more insights especially for n_classes>2
        self.val_confmat = ConfusionMatrix(task=task, num_classes=self.net.n_classes)
        self.test_confmat = ConfusionMatrix(task=task, num_classes=self.net.n_classes)

        # Visualization
        self.illustrator = illustrator

    def forward(self, x: torch.Tensor, y: torch.Tensor=None, k: List=None):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_best_auroc.reset()
        self.val_best_acc.reset()

    def step(self, batch: Any):
        features, targets, keys = batch

        logits, Y_prob, Y_hat, attention, results_dict = self.forward(x=features) # n X L
        attention = {
            "keys": keys,
            "scores": attention
            }

        loss = self.slide_loss_fn(logits, targets)
        ## Self-distillation Losses:
        f_t = torch.mean(results_dict['features_teacher'], 1).squeeze()
        f_s = results_dict['features_student'].squeeze()
        loss_coefficient = 0.3
        feature_loss_coefficient = 0.03
        loss += self.slide_loss_fn(results_dict['student_logits'], targets)*(1 - loss_coefficient)
        loss += CrossEntropy(results_dict['student_logits'], logits)*loss_coefficient
        ## Teacher becomes student and vice versa
        loss += torch.dist(f_s, f_t) * feature_loss_coefficient

        return loss, logits, Y_prob, Y_hat, targets, attention

    def training_step(self, batch: Any, batch_idx: int):
        loss, logits, Y_prob, Y_hat, targets, attention = self.step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"loss": loss, "preds": Y_prob, "targets": targets}
    
    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        preds = torch.cat([output['preds'] for output in outputs], dim = 0).squeeze()
        targets = torch.stack([output['targets'] for output in outputs], dim = 0).squeeze()
        self.train_metrics(preds , targets)
        self.log_dict(self.train_metrics, on_step=False, on_epoch = True, prog_bar=True, sync_dist=True)

    def validation_step(self, batch: Any, batch_idx: int):
        loss, logits, Y_prob, Y_hat, targets, attention = self.step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"loss": loss, "preds": Y_prob, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        preds = torch.cat([output['preds'] for output in outputs], dim = 0).squeeze()
        targets = torch.stack([output['targets'] for output in outputs], dim = 0).squeeze()
        self.val_metrics(preds , targets)
        self.log_dict(self.val_metrics, on_step=False, on_epoch = True, prog_bar=True, sync_dist=True)

        metrics = self.val_metrics.compute()  # get current val acc
        self.val_best_auroc(metrics["val/MulticlassAUROC"])  # update best so far val acc
        self.val_best_acc(metrics["val/MulticlassAccuracy"])
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/AUROC_best", self.val_best_auroc.compute(), prog_bar=True, sync_dist=True)
        self.log("val/acc_best", self.val_best_acc.compute(), prog_bar=True, sync_dist=True)
        # Add Confusion Matrix to Tensorboard
        tb = self.loggers[1].experiment       
        self.val_confmat(preds , targets)
        cm = self.val_confmat.compute()
        cm = plot_confmat(confmat=cm.cpu().numpy(), num_classes=self.net.n_classes)
        tb.add_image("val/ConfusionMatrix", cm, global_step=self.current_epoch)
        # Reset metrics
        self.val_confmat.reset()
    

    def test_step(self, batch: Any, batch_idx: int):
        loss, logits, Y_prob, Y_hat, targets, attention = self.step(batch)
        if self.illustrator:
            self.illustrator.save_attention_maps(prediction=Y_hat, target=targets, attention=attention)
        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"loss": loss, "preds": Y_prob, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        preds = torch.cat([output['preds'] for output in outputs], dim = 0).squeeze()
        targets = torch.stack([output['targets'] for output in outputs], dim = 0).squeeze()
        self.test_metrics(preds , targets)
        self.log_dict(self.test_metrics, on_step=False, on_epoch = True, prog_bar=True, sync_dist=True)
        # Add Confusion Matrix to Tensorboard
        tb = self.loggers[1].experiment
        self.test_confmat(preds , targets)
        cm = self.test_confmat.compute()
        cm = plot_confmat(confmat=cm.cpu().numpy(), num_classes=self.net.n_classes)
        tb.add_image("test/ConfusionMatrix", cm)
        

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        if type(self.hparams.optimizer) is DictConfig:
            optimizer = create_optimizer(self.hparams.optimizer, self.net)
        
        else:
            optimizer = self.hparams.optimizer(params=self.parameters())

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "clam.yaml")
    _ = hydra.utils.instantiate(cfg)
