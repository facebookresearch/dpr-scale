#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Optional
import torch.nn as nn
from dpr_scale.utils.utils import PathManager

# @manual=//python/wheel/transformers3:transformers3
from transformers import AutoModel, AutoConfig


class ColBERTEncoder(nn.Module):
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
        cfg.output_hidden_states = True
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
                linear,
            )

    def forward(self, tokens, **kwargs):
        # make it transformer 4.x compatible
        outputs = self.transformer(**tokens, return_dict=True)  # B x T x C
        hiddens = self.project(outputs.hidden_states[-1][:, 1:, :])
        attention_mask = tokens['attention_mask'][:, 1:].unsqueeze(-1)
        expert_repr = attention_mask * hiddens
        return {"expert_repr": expert_repr.clone()}
