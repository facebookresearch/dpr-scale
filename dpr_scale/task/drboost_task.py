#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import hydra
import torch
from dpr_scale.utils.utils import ScriptMultiEncoder
from dpr_scale.task.dpr_task import DenseRetrieverTask
from typing import List


# Implementation of https://arxiv.org/pdf/2112.07771.pdf.
# Primarily used for inference by combining all weak encoders.
class DrBoostTask(DenseRetrieverTask):
    def __init__(
        self,
        checkpoint_paths: List[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.checkpoint_paths = checkpoint_paths

    def setup(self, stage: str):
        # skip building model during test.
        if stage == "test" and self.setup_done:
            return
        # load all weak encoders from the provided checkpoints.
        self.weak_encoders = torch.nn.ModuleList()
        for idx, ckpt_path in enumerate(self.checkpoint_paths):
            weak_encoder = DenseRetrieverTask.load_from_checkpoint(ckpt_path, map_location=self.device)
            assert weak_encoder.device == self.device
            self.weak_encoders.append(weak_encoder)
            print(f"Loaded weak encoder #{idx} state dict from {ckpt_path} ...")
        self.setup_done = True

    def forward(self, query_ids, contexts_ids):
        for weak_encoder in self.weak_encoders:
            weak_encoder.to(self.device)
            assert weak_encoder.device == self.device
        # encode query and contexts
        query_repr = self.encode_queries(query_ids)  # bs x (n_enc * d)
        contexts_repr = self.encode_contexts(contexts_ids)  # ctx_cnt x (n_enc * d)
        return query_repr, contexts_repr

    def configure_optimizers(self):
        pass

    def encode_queries(self, query_ids):
        all_query_list = []
        for weak_encoder in self.weak_encoders:
            all_query_list.append(self._encode_sequence(query_ids, weak_encoder.query_encoder))  # bs x d
        query_repr = torch.cat(all_query_list, dim=1)  # bs * (n_enc * d)
        return query_repr

    def encode_contexts(self, context_ids):
        all_context_list = []
        for weak_encoder in self.weak_encoders:
            all_context_list.append(self._encode_sequence(context_ids, weak_encoder.context_encoder))  # bs x d
        context_repr = torch.cat(all_context_list, dim=1)  # bs * (n_enc * d)
        return context_repr

    @torch.no_grad()
    def to_torchscript(
        self,
        file_path=None,
        method="script",
        example_inputs=None,
        **kwargs,
    ):
        mode = self.training
        if method == "script":
            transform = hydra.utils.instantiate(self.transform_conf)
            ctx_encoder = ScriptMultiEncoder(transform, [enc.context_encoder for enc in self.weak_encoders])
            ctx_encoder = torch.jit.script(ctx_encoder.eval(), **kwargs)
            result = {"ctx_encoder": ctx_encoder}
            # Quantize. TODO when PL has better handling link this with the save_quantized
            # flag in ModelCheckpoint
            ctx_encoder_qt = ScriptMultiEncoder(transform, [enc.context_encoder for enc in self.weak_encoders], True)
            ctx_encoder_qt = torch.jit.script(ctx_encoder_qt.eval(), **kwargs)
            result["ctx_encoder_qt"] = ctx_encoder_qt
            if not self.shared_model:
                q_encoder = ScriptMultiEncoder(transform, [enc.query_encoder for enc in self.weak_encoders])
                q_encoder = torch.jit.script(q_encoder.eval(), **kwargs)
                result["q_encoder"] = q_encoder
                # Quantize. TODO when PL has better handling link this with the save_quantized
                # flag in ModelCheckpoint
                q_encoder_qt = ScriptMultiEncoder(transform, [enc.query_encoder for enc in self.weak_encoders], quantize=True)
                q_encoder_qt = torch.jit.script(q_encoder_qt.eval(), **kwargs)
                result["q_encoder_qt"] = q_encoder_qt
        else:
            raise ValueError(
                "The 'method' parameter only supports 'script',"
                f" but value given was: {method}"
            )

        self.train(mode)

        if file_path is not None:
            torch.jit.save(ctx_encoder, file_path)

        return result
