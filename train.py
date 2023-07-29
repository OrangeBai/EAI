import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from config import *
from core.pl_model import BaseModel, ForwardRecorderModel
from settings.train_parser import TrainParser

if __name__ == "__main__":
    args = TrainParser().get_args()
    model_dir = os.path.join(MODEL_PATH, args.dataset, args.net, args.project)
    os.makedirs(model_dir, exist_ok=True)

    logtool = WandbLogger(name=args.name, save_dir=model_dir, project=args.project, config=args)
    if args.mode == "std":
        model = BaseModel(args)
    elif args.mode == "preact":
        model = ForwardRecorderModel(args)
    else:
        raise NameError("aaa")

    callbacks = [
        ModelCheckpoint(
            monitor="val/top1",
            save_top_k=1,
            mode="max",
            save_on_train_epoch_end=False,
            dirpath=logtool.experiment.dir,
            filename="best",
        ),
    ]

    trainer = pl.Trainer(
        devices="auto",
        precision=32,
        amp_backend="native",
        accelerator="cuda",
        strategy="dp",
        # callbacks=callbacks,
        max_epochs=args.num_epoch,
        logger=logtool,
        enable_progress_bar=args.npbar,
        inference_mode=False,
    )
    trainer.fit(model)
    model.save_model()
