import os
from argparse import Namespace

import pytorch_lightning as pl
import torch

from core.dataloader import set_dataloader
from core.utils import accuracy, init_optimizer, init_scheduler, normal_init, apply_init
from models import build_model
from models.blocks import ConvBlock, LinearBlock


class BaseModel(pl.LightningModule):
    def __init__(self, args: Namespace):
        # model, dataset, batch_size, num_workers, act, optimizer, lr, lr_scheduler):
        """
        init base trainer
        """
        super().__init__()
        self.args = args
        self.model = build_model(args.net, args.dataset, args.act, args.bn)
        self._init_weight()
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.train_loader, self.val_loader = set_dataloader(
            args.dataset, batch_size=args.batch_size, num_workers=args.num_workers
        )

    def _init_weight(self):
        if self.args.act.lower() not in ["sigmoid", "tanh", "relu", "selu", "leaky_relu"]:
            non_linearity = "relu"
        else:
            non_linearity = self.args.act.lower()
        for m in self.model.modules():
            apply_init(m, non_linearity) if self.args.init else normal_init(m)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def configure_optimizers(self):
        num_step = self.trainer.max_epochs * len(self.train_loader)
        optimizer = init_optimizer(self.model, self.args.optimizer, lr=self.args.lr)

        lr_scheduler = init_scheduler(self.args.lr, self.args.lr_scheduler, num_step, optimizer=optimizer)
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def training_step(self, batch, batch_idx):
        images, labels = batch[0], batch[1]
        outputs = self.model(images)
        loss = self.loss_function(outputs, labels)
        top1, top5 = accuracy(outputs, labels)

        self.log("train/loss", loss, sync_dist=True)
        self.log("train/top1", top1, sync_dist=True)
        self.log("lr", self.lr, sync_dist=True)
        return loss

    @property
    def lr(self):
        return self.optimizers().optimizer.param_groups[0]["lr"]

    def validation_step(self, batch, batch_idx):
        images, labels = batch[0], batch[1]
        pred = self.model(images)
        top1, top5 = accuracy(pred, labels)
        loss = self.loss_function(pred, labels)
        self.log("val/loss", loss, sync_dist=True, on_step=False, on_epoch=True)
        self.log("val/top1", top1, sync_dist=True, on_step=False, on_epoch=True)
        return

    def valid_blocks(self):
        for name, block in self.named_modules():
            if self.check_valid(block):
                yield name, block

    def check_valid(self, block):
        return block != self.model.layers[-1] and isinstance(block, (ConvBlock, LinearBlock))

    def save_model(self):
        exp = getattr(self.logger, "experiment")
        torch.save(self.model, os.path.join(exp.dir, "model.pth"))
