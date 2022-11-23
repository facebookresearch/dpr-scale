#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import pickle
import torch
from dpr_scale.task.cross_encoder_task import CrossEncoderTask
from dpr_scale.utils.utils import PathManager


class RerankCrossEncoderTask(CrossEncoderTask):
    def __init__(
        self,
        output_dir,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True) 
    
    def _eval_step(self, batch, batch_idx):
        token_ids = batch["text_ids"]  # bs x q_cnt x q_len
        scores = self(token_ids)
        if len(scores.shape) > 1 and scores.shape[-1] > 1:
            scores = scores.max(1).values
        return [batch["qid"], batch["ctx_id"], scores.cpu()]

    def training_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx)
    
    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx)
    
    def test_epoch_end(self, test_outputs):
        qids, ctx_ids, scores = [], [], []
        for entry in test_outputs:
            b_qids, b_ctx_ids, b_scores = entry
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
