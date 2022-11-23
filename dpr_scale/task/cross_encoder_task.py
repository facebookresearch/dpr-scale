#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import hydra
import torch
from dpr_scale.utils.utils import PathManager
from pytorch_lightning import LightningModule
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import (
    fp16_compress_hook,
)
from torch.serialization import default_restore_location

# This module is implemented only for inference; Training is not implemented
class CrossEncoderTask(LightningModule):
    def __init__(
        self,
        transform,
        model,
        datamodule,
        optim,
        k=1,  # k for accuracy@k metric
        shared_model: bool = True,  # shared encoders
        in_batch_eval: bool = True,  # use only in-batch contexts for val
        warmup_steps: int = 0,
        fp16_grads: bool = False,
        pretrained_checkpoint_path: str = "",
    ):
        super().__init__()
        # save all the task hyperparams
        # so we can instantiate it much easily later.
        self.save_hyperparameters()
        self.transform_conf = (
            transform.text_transform
            if hasattr(transform, "text_transform")
            else transform
        )
        self.model_conf = model
        self.k = k
        self.fp16_grads = fp16_grads
        self.pretrained_checkpoint_path = pretrained_checkpoint_path
        self.setup_done = False

    def setup(self, stage: str):
        # skip building model during test.
        # Otherwise, the state dict will be re-initialized
        if stage == "test" and self.setup_done:
            return
        # resetting call_configure_sharded_model_hook attribute so that we could configure model
        self.call_configure_sharded_model_hook = False

        self.cross_encoder = hydra.utils.instantiate(
            self.model_conf,
        )
        if self.pretrained_checkpoint_path:
            checkpoint_dict = torch.load(
                PathManager.open(self.pretrained_checkpoint_path, "rb"),
                map_location=lambda s, l: default_restore_location(s, "cpu"),
            )
            self.load_state_dict(checkpoint_dict["state_dict"])
            print(f"Loaded state dict from {self.pretrained_checkpoint_path}")

        self.setup_done = True
    
    def on_load_checkpoint(self, checkpoint) -> None:
        """
        This hook will be called before loading state_dict from a checkpoint.
        setup("fit") will built the model before loading state_dict
        """
        self.setup("fit")

    def on_pretrain_routine_start(self):
        if self.fp16_grads:
            self.trainer.strategy._model.register_comm_hook(None, fp16_compress_hook)
    
    def configure_optimizers(self):
        pass

    def forward(self, token_ids):
        # encode query and contexts
        scores = self.cross_encoder(token_ids)  # bs x d
        return scores
