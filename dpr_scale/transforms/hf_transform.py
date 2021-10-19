#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch.nn as nn
from dpr_scale.utils.utils import PathManager

# @manual=//python/wheel/transformers3:transformers3
from transformers import AutoTokenizer


class HFTransform(nn.Module):
    def __init__(
        self,
        model_path: str = "roberta-base",
        max_seq_len: int = 256,
        add_special_tokens: bool = True,
        return_tensors: bool = True,
    ):
        super().__init__()
        local_model_path = PathManager.get_local_path(model_path, recursive=True)
        self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        self.sep_token = self.tokenizer.sep_token
        self.max_seq_len = max_seq_len
        self.add_special_tokens = add_special_tokens
        self.return_tensors = return_tensors

    def forward(self, texts):
        return self.tokenizer(
            texts,
            return_tensors="pt" if self.return_tensors else None,
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
            add_special_tokens=self.add_special_tokens,
        )
