#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import pickle
import torch
from dpr_scale.task.dpr_task import DenseRetrieverTask
from dpr_scale.utils.utils import PathManager
from pytorch_lightning.utilities.cloud_io import load as pl_load


class RerankDenseRetrieverTask(DenseRetrieverTask):
    def __init__(
        self,
        checkpoint_path,
        output_dir,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.checkpoint_path = checkpoint_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True) 
    
    def setup(self, stage: str):
        super().setup("train")
        print(f"Loading checkpoint from {self.checkpoint_path}")
        checkpoint = pl_load(
            self.checkpoint_path, map_location=lambda storage, loc: storage)
        self.load_state_dict(checkpoint['state_dict'])
    
    def forward(self, query_ids, ctx_ids):
        # encode questions
        query_repr = self.encode_queries(query_ids)  # q_cnt x d
        ctx_repr = self.encode_contexts(ctx_ids)
        return query_repr, ctx_repr

    def _eval_step(self, batch, batch_idx):
        q_ids = batch["query_ids"]  # bs x q_cnt x q_len
        ctx_ids = batch["contexts_ids"]
        q_repr, ctx_repr = self(q_ids, ctx_ids)
        scores = torch.sum(q_repr * ctx_repr, 1)
        return [batch["qid"], batch["ctx_id"], scores.cpu()]
    
    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx)

    def test_epoch_end(self, test_outputs):
        qids, ctx_ids, scores = [], [], []
        for b_qids, b_ctx_ids, b_scores in test_outputs:
            qids.extend(b_qids)
            ctx_ids.extend(b_ctx_ids)
            scores.append(b_scores)
        scores = torch.cat(scores, dim=0)
        out_file = os.path.join(
            self.output_dir, f"scores_{self.global_rank:04}.pkl")
        
        print(f"\nWriting scores to {out_file}")
        with PathManager.open(out_file, mode="wb") as f:
            pickle.dump(scores, f, protocol=4)

        out_file = os.path.join(
            self.output_dir, f"qids_{self.global_rank:04}.pkl")
        with PathManager.open(out_file, mode="wb") as f:
            pickle.dump(qids, f, protocol=4)

        out_file = os.path.join(
            self.output_dir, f"ctx_ids_{self.global_rank:04}.pkl")
        with PathManager.open(out_file, mode="wb") as f:
            pickle.dump(ctx_ids, f, protocol=4)

        torch.distributed.barrier()  # make sure rank 0 waits for all to complete