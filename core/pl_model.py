import os
import random
from argparse import Namespace

import numpy as np
import pytorch_lightning as pl
import torch
import wandb

from core.dataloader import set_dataloader
from core.pattern import PreActHook
from core.utils import accuracy, init_optimizer, init_scheduler, normal_init, apply_init
from models import build_model
from models.blocks import ConvBlock, LinearBlock, BasicBlock, Bottleneck


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
        num_step = self.trainer.max_epochs * len(self.train_loader) - self.global_step
        optimizer = init_optimizer(self.model, self.args.optimizer, lr=self.args.lr, args=self.args)

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
        return block != self.model.layers[-1] and isinstance(block, (ConvBlock, LinearBlock, BasicBlock, Bottleneck))

    def save_model(self):
        exp = getattr(self.logger, "experiment")
        torch.save(self.model, os.path.join(exp.dir, "model.pth"))

    def on_before_backward(self, loss) -> None:
        if self.global_step < 500:
            for param_group in self.optimizers().optimizer.param_groups:
                param_group['lr'] = self.args.lr * 0.01
        elif self.global_step == 500:
            for idx, param_group in enumerate(self.optimizers().optimizer.param_groups):
                param_group['lr'] = self.lr_schedulers()._get_closed_form_lr()[idx]
        else:
            return


class ForwardRecorderModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.forward_hook = PreActHook(self)

    # def on_train_start(self) -> None:
    #     wandb.watch(self.model, log="all", log_freq=500)

    def training_step(self, batch, batch_idx):
        if self.global_step % 200 == 0:
            self.forward_hook.set_up()
            loss = super().training_step(batch, batch_idx)
            idx = random.randint(0, self.args.batch_size - 1)
            features = self.forward_hook.retrieve()
            mean = np.array([feature[idx].mean() for feature in features])
            var = np.array([feature[idx].var(axis=0).mean() for feature in features])

            wandb.log({"pre_act/mean": mean, "pre_act/var": var}, step=self.global_step)
            self.forward_hook.remove()
        else:
            loss = super().training_step(batch, batch_idx)
        return loss

    def on_after_backward(self) -> None:
        if self.global_step % 100 != 0:
            return
        weights = {name: self.load_block_weight(block) for name, block in self.valid_blocks()}
        gradients = {name: self.load_block_gradient(block) for name, block in self.valid_blocks()}
        ratios = {name: abs(grad) / (abs(weight) + 1e-5) for (name, weight), (_, grad) in
                  zip(weights.items(), gradients.items())}

        weights_mean = {name: weight.mean() for name, weight in weights.items()}
        weights_var = {name: weight.var() for name, weight in weights.items()}

        grad_mean = {name: grad.mean() for name, grad in gradients.items()}
        grad_var = {name: grad.var() for name, grad in gradients.items()}

        ratio_mean = {name: ratio.mean() for name, ratio in ratios.items()}
        ratio_var = {name: ratio.var() for name, ratio in ratios.items()}

        weights_hist = {'weight/' + name: wandb.Histogram(weight) for name, weight in weights.items()}
        grad_hist = {'grad/' + name: wandb.Histogram(grad) for name, grad in gradients.items()}
        ratio_hist = {'ratio/' + name: wandb.Histogram(ratio) for name, ratio in ratios.items()}

        wandb.log(
            {
                "weights_mean": list(weights_mean.values()),
                "weights_var": list(weights_var.values()),
                "grad_mean": list(grad_mean.values()),
                "grad_var": list(grad_var.values()),
                "ratio_mean": list(ratio_mean.values()),
                "ratio_var": list(ratio_var.values()),
                **weights_hist,
                **grad_hist,
                **ratio_hist
            },
            step=self.global_step
        )
        return

    @staticmethod
    def load_block_weight(block):
        if isinstance(block, (LinearBlock, ConvBlock)):
            return block.LT.weight.detach().cpu()
        if isinstance(block, (Bottleneck, BasicBlock)):
            return block.conv2.weight.detach().cpu()

    @staticmethod
    def load_block_gradient(block):
        if isinstance(block, (LinearBlock, ConvBlock)):
            return block.LT.weight.grad.detach().cpu()
        if isinstance(block, (Bottleneck, BasicBlock)):
            return block.conv2.weight.grad.detach().cpu()
