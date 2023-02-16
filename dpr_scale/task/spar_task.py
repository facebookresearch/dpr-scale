#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import hydra
import torch
from dpr_scale.utils.utils import ScriptMultiEncoder
from dpr_scale.task.dpr_task import DenseRetrieverTask

# This Task saves a unified spar checkpoint by loading a pre-trained dpr checkpoint
# and a lexical model checkpoint. Both checkpoints need to be saved with hyperparameters.
class SalientPhraseAwareDenseRetrieverTask(DenseRetrieverTask):
    def __init__(
        self,
        pretrained_checkpoint_path: str = "",
        lexical_model_checkpoint_path: str = "",
        lexical_weight: float = 0,
        **kwargs,
    ):
        # init dpr model
        super().__init__(**kwargs)
        self.pretrained_checkpoint_path = pretrained_checkpoint_path
        self.lexical_model_checkpoint_path = lexical_model_checkpoint_path
        self.lexical_weight = lexical_weight

    def setup(self, stage: str):
        # skip building model during test.
        # Otherwise, the state dict will be re-initialized
        if stage == "test" and self.setup_done:
            return

        self.lexical_model = DenseRetrieverTask.load_from_checkpoint(self.lexical_model_checkpoint_path, map_location=self.device)
        self.dense_model = DenseRetrieverTask.load_from_checkpoint(self.pretrained_checkpoint_path, map_location=self.device)
        self.setup_done = True

    def encode_queries(self, query_ids):
        query_repr = self._encode_sequence(
            query_ids, self.dense_model.query_encoder
        )  # bs x d
        # lexical model representations
        lex_query_repr = self.dense_model._encode_sequence(
            query_ids, self.lexical_model.query_encoder
        )
        lex_query_repr *= self.lexical_weight
        query_repr = torch.cat([query_repr, lex_query_repr], dim=-1)
        return query_repr

    def encode_contexts(self, contexts_ids):
        contexts_repr = self._encode_sequence(
            contexts_ids, self.dense_model.context_encoder
        )  # ctx_cnt x d
        # lexical model representations
        lex_contexts_repr = self._encode_sequence(
            contexts_ids, self.lexical_model.context_encoder
        )
        # lexical weight is only applied to the queries.
        contexts_repr = torch.cat([contexts_repr, lex_contexts_repr], dim=-1)

        return contexts_repr

    def training_step(self, batch, batch_idx):
        # we don't use training for SPAR
        return super().training_step(batch, batch_idx)

    @torch.no_grad()
    def to_torchscript(
        self,
        file_path=None,
        method="script",
        example_inputs=None,
        **kwargs,
    ):
        # use ScriptMultiEncoder for SPAR torchscirpting
        mode = self.training
        if method == "script":
            encoders = [self.dense_model, self.lexical_model]
            transform = hydra.utils.instantiate(self.transform_conf)
            ctx_encoder = ScriptMultiEncoder(transform, [enc.context_encoder for enc in encoders])
            ctx_encoder = torch.jit.script(ctx_encoder.eval(), **kwargs)
            result = {"ctx_encoder": ctx_encoder}
            # Quantize. TODO when PL has better handling link this with the save_quantized
            # flag in ModelCheckpoint
            ctx_encoder_qt = ScriptMultiEncoder(transform, [enc.context_encoder for enc in encoders], True)
            ctx_encoder_qt = torch.jit.script(ctx_encoder_qt.eval(), **kwargs)
            result["ctx_encoder_qt"] = ctx_encoder_qt
            if not self.shared_model:
                q_encoder = ScriptMultiEncoder(transform, [enc.query_encoder for enc in encoders], False,
                    weights=[1., self.lexical_weight])
                q_encoder = torch.jit.script(q_encoder.eval(), **kwargs)
                result["q_encoder"] = q_encoder
                # Quantize. TODO when PL has better handling link this with the save_quantized
                # flag in ModelCheckpoint
                q_encoder_qt = ScriptMultiEncoder(transform, [enc.query_encoder for enc in encoders], quantize=True,
                    weights=[1., self.lexical_weight])
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
