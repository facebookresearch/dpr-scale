#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import pathlib
import pickle
import torch
from dpr_scale.task.dpr_task import DenseRetrieverTask
from dpr_scale.utils.utils import PathManager


class GenerateEmbeddingsJitTask(DenseRetrieverTask):
    def __init__(self, ctx_embeddings_dir, jit_checkpoint_path, **kwargs):
        super().__init__(**kwargs)
        self.ctx_embeddings_dir = ctx_embeddings_dir
        self.jit_checkpoint_path = jit_checkpoint_path
        pathlib.Path(ctx_embeddings_dir).mkdir(parents=True, exist_ok=True)

    def setup(self, stage: str):
        print(f"Loading checkpoint from {self.jit_checkpoint_path} ...")
        local_path = PathManager.get_local_path(self.jit_checkpoint_path)
        self.jit_model = torch.jit.load(local_path)
        self.jit_model.to(self.device)

    def forward(self, contexts_ids):
        # encode contexts
        contexts_repr = self.jit_model.encode(contexts_ids)  # ctx_cnt x d
        return contexts_repr

    def _eval_step(self, batch, batch_idx):
        contexts_ids = batch["contexts_ids"]  # bs x ctx_cnt x ctx_len
        contexts_repr = self(contexts_ids)
        return contexts_repr.cpu()

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx)

    def test_epoch_end(self, contexts_repr):
        contexts_repr = torch.cat(contexts_repr, dim=0)
        if not self.ctx_embeddings_dir:
            self.ctx_embeddings_dir = self.trainer.weights_save_path
        out_file = os.path.join(
            self.ctx_embeddings_dir, f"reps_{self.global_rank:04}.pkl")
        print(f"\nWriting tensor of size {contexts_repr.size()} to {out_file}")
        with PathManager.open(out_file, mode="wb") as f:
            pickle.dump(contexts_repr, f, protocol=4)
        if torch.distributed.is_initialized():
            torch.distributed.barrier() # make sure rank 0 waits for all to complete


class GenerateQueryEmbeddingsJitTask(GenerateEmbeddingsJitTask):
    def __init__(
        self,
        query_emb_output_dir=None,
        passages="",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.query_emb_output_dir = query_emb_output_dir or self.ctx_embeddings_dir

    def forward(self, query_ids):
        # encode questions
        query_repr = self.jit_model.encode(query_ids)  # q_cnt x d
        return query_repr

    def _eval_step(self, batch, batch_idx):
        q_ids = batch["query_ids"]  # bs x q_cnt x q_len
        q_repr = self(q_ids)
        return q_repr.cpu()

    def test_epoch_end(self, queries_repr):
        queries_repr = torch.cat(queries_repr, dim=0)
        out_file = self.query_emb_output_dir
        pathlib.Path(self.query_emb_output_dir).parent.mkdir(parents=True, exist_ok=True)
        out_file = os.path.join(
            self.query_emb_output_dir, f"qry_reps_{self.global_rank:04}.pkl")
        print(f"\nWriting tensor of size {queries_repr.size()} to {out_file}")
        with PathManager.open(out_file, mode="wb") as f:
            pickle.dump(queries_repr, f, protocol=4)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
