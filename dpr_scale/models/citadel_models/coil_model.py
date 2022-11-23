#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Optional
import torch.nn as nn
from dpr_scale.utils.utils import PathManager

# @manual=//python/wheel/transformers3:transformers3
from transformers import AutoModel, AutoConfig


class COILEncoder(nn.Module):
    def __init__(
        self,
        model_path: str = "roberta-base",
        dropout: float = 0.1,
        projection_dim: Optional[int] = None,
        cls_projection_dim: Optional[int] = None,
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
        
        self.cls_project = nn.Identity()  # to make torchscript happy
        if cls_projection_dim:
            linear = nn.Linear(cfg.hidden_size, cls_projection_dim)
            linear.weight.data.normal_(mean=0.0, std=0.02)
            self.cls_project = nn.Sequential(
                linear,)

    def forward(self, tokens, add_cls=False, **kwargs):
        ret = {}
        # make it transformer 4.x compatible
        outputs = self.transformer(**tokens, return_dict=True)  # B x T x C
        hiddens = self.project(outputs.hidden_states[-1][:, 1:, :])
        attention_mask = tokens['attention_mask'][:, 1:].unsqueeze(-1)
        expert_repr = attention_mask * hiddens
        
        if add_cls:
            cls_repr = self.cls_project(outputs.hidden_states[-1][:, 0, :])
            ret["cls_repr"] = cls_repr.clone()

        ret["expert_repr"] = expert_repr.clone()
        ret["expert_ids"] = tokens["input_ids"][:, 1:].clone()
        ret["expert_weights"] = tokens['attention_mask'][:, 1:].clone()
        ret["attention_mask"] = tokens["attention_mask"][:, 1:].clone()
        return ret
