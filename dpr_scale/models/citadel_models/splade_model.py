#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch.nn as nn
import torch
from dpr_scale.utils.utils import PathManager

# @manual=//python/wheel/transformers3:transformers3
from transformers import AutoModelForMaskedLM, AutoConfig


class SPLADEEncoder(nn.Module):
    def __init__(
        self,
        model_path: str = "roberta-base",
        dropout: float = 0.1,
    ):
        super().__init__()
        # remove recursive argument which is not supported now
        local_model_path = PathManager.get_local_path(model_path)
        cfg = AutoConfig.from_pretrained(local_model_path)
        cfg.attention_probs_dropout_prob = dropout
        cfg.hidden_dropout_prob = dropout
        self.transformer = AutoModelForMaskedLM.from_pretrained(local_model_path, config=cfg)

    def forward(self, tokens):
        # make it transformer 4.x compatible
        outputs = self.transformer(**tokens, return_dict=True)  # B x T x C
        p_logits = outputs.logits[:, 1:, :]
        attention_mask = tokens['attention_mask'][:, 1:].unsqueeze(-1)
        sentence_rep = torch.max(torch.log(1 + torch.relu(p_logits)) * attention_mask, dim=1).values
        return sentence_rep.clone()
