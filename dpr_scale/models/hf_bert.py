#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch.nn as nn
from dpr_scale.utils.utils import PathManager

# @manual=//python/wheel/transformers3:transformers3
from transformers import BertModel, BertConfig


class BertEncoder(nn.Module):
    def __init__(
        self,
        model_path: str = "bert-base-uncased",
        dropout: float = 0.1,
    ):
        super().__init__()
        # remove recursive argument which is not supported now
        local_model_path = PathManager.get_local_path(model_path)
        cfg = BertConfig.from_pretrained(local_model_path)
        cfg.attention_probs_dropout_prob = dropout
        cfg.hidden_dropout_prob = dropout
        self.transformer = BertModel.from_pretrained(local_model_path, config=cfg)

    def forward(self, tokens):
        last_layer, _ = self.transformer(**tokens)  # B x T x C
        sentence_rep = last_layer[:, 0, :]
        return sentence_rep.clone()  # to force into overlapping memory
