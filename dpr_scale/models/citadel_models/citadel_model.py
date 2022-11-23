#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Optional
import torch.nn as nn
import torch
from dpr_scale.utils.utils import PathManager

# @manual=//python/wheel/transformers3:transformers3
from transformers import AutoModelForMaskedLM, AutoConfig

class CITADELEncoder(nn.Module):
    def __init__(
        self,
        model_path: str = "bert-base-uncased",
        dropout: float = 0.1,
        tok_projection_dim: Optional[int] = None,
        cls_projection_dim: Optional[int] = None,
    ):
        super().__init__()
        # remove recursive argument which is not supported now
        local_model_path = PathManager.get_local_path(model_path)
        cfg = AutoConfig.from_pretrained(local_model_path)
        cfg.output_hidden_states = True
        cfg.attention_probs_dropout_prob = dropout
        cfg.hidden_dropout_prob = dropout
        cfg.tok_projection_dim = tok_projection_dim
        self.transformer = AutoModelForMaskedLM.from_pretrained(local_model_path, config=cfg)
        
        self.cls_project = nn.Identity()  # to make torchscript happy
        if cls_projection_dim:
            linear = nn.Linear(cfg.hidden_size, cls_projection_dim)
            linear.weight.data.normal_(mean=0.0, std=0.02)
            self.cls_project = nn.Sequential(
                linear,
            )

        self.tok_project = nn.Identity()  # to make torchscript happy
        if tok_projection_dim:
            linear = nn.Linear(cfg.hidden_size, tok_projection_dim)
            linear.weight.data.normal_(mean=0.0, std=0.02)
            self.tok_project = nn.Sequential(
                linear,
            )

    def forward(self, tokens, topk=1, add_cls=False):
        ret = {}
        # make it transformer 4.x compatible
        outputs = self.transformer(**tokens, return_dict=True)
        hiddens = outputs.hidden_states[-1][:, 1:, :]
        attention_mask = tokens["attention_mask"][:, 1:]
        logits = outputs.logits[:, 1:, :]
        
        # router representation
        full_router_repr = torch.log(1 + torch.relu(logits)) * attention_mask.unsqueeze(-1)
        router_repr = torch.max(full_router_repr, dim=1).values
        # routing, assign every token to top-k expert
        expert_weights, expert_ids = torch.topk(full_router_repr, dim=2, k=topk) # B x T x topk
        # expert representation
        expert_repr = self.tok_project(hiddens) * attention_mask.unsqueeze(-1)
        
        # some training stat.
        router_mask = torch.zeros_like(full_router_repr)
        router_mask.scatter_(dim=2, index=expert_ids, src=(expert_weights > 0.).to(expert_weights.dtype)) # B x T x E 
        # average number of experts per input 
        ret["avg_cond_num_experts"] = router_mask.sum(1).sum(1, keepdim=True).mean(0, keepdim=True)
        # average number of distinct experts per batch
        ret["avg_marg_num_experts"] = router_mask.sum(1).max(0, keepdim=True).values.sum(1, keepdim=True)

        # for computing load balancing
        router_softmax_repr = torch.softmax(logits, dim=-1)
        ret["router_mask"] = router_mask.sum(1).clone()
        ret["router_softmax_repr"] = router_softmax_repr.sum(1).clone()
        ret["attention_mask"] = attention_mask.clone()
        if add_cls:
            cls_repr = self.cls_project(outputs.hidden_states[-1][:, 0, :])
            ret["cls_repr"] = cls_repr.clone()
        ret["router_repr"] = router_repr.clone()
        ret["expert_ids"] = expert_ids.clone()
        ret["expert_repr"] = expert_repr.clone()
        ret["expert_weights"] = expert_weights.clone()
        return ret
