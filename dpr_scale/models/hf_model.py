#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Optional
import torch.nn as nn
from dpr_scale.utils.utils import PathManager

# @manual=//python/wheel/transformers3:transformers3
from transformers import AutoModel, AutoConfig


class HFEncoder(nn.Module):
    def __init__(
        self,
        model_path: str = "roberta-base",
        dropout: float = 0.1,
        projection_dim: Optional[int] = None,
    ):
        super().__init__()
        # remove recursive argument which is not supported now
        local_model_path = PathManager.get_local_path(model_path)
        cfg = AutoConfig.from_pretrained(local_model_path)
        cfg.attention_probs_dropout_prob = dropout
        cfg.hidden_dropout_prob = dropout
        self.transformer = AutoModel.from_pretrained(local_model_path, config=cfg)
        self.project = nn.Identity()  # to make torchscript happy
        if projection_dim == -1:
            projection_dim = cfg.hidden_size
        if projection_dim:
            linear = nn.Linear(cfg.hidden_size, projection_dim)
            linear.weight.data.normal_(mean=0.0, std=0.02)
            self.project = nn.Sequential(
                linear, nn.LayerNorm(projection_dim)
            )

    def forward(self, tokens):
        # make it transformer 4.x compatible
        outputs = self.transformer(**tokens)  # B x T x C
        last_layer = outputs[0]
        sentence_rep = self.project(last_layer[:, 0, :])
        return sentence_rep.clone()
