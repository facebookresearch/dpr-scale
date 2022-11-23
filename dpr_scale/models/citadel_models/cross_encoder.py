#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch.nn as nn
import torch
from dpr_scale.utils.utils import PathManager

# @manual=//python/wheel/transformers3:transformers3
from transformers import AutoModelForSequenceClassification

class CrossEncoder(nn.Module):
    def __init__(
        self,
        model_path: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        super().__init__()
        # remove recursive argument which is not supported now
        local_model_path = PathManager.get_local_path(model_path)
        self.transformer = AutoModelForSequenceClassification.from_pretrained(local_model_path)

    def forward(self, tokens):
        # make it transformer 4.x compatible
        with torch.no_grad():
            outputs = self.transformer(**tokens, return_dict=True)  # B x T x C
            scores = outputs.logits
        return scores
