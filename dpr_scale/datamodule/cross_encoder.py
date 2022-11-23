#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from dpr_scale.utils.utils import (
    ContiguousDistributedSamplerForTest,
    maybe_add_title,
)
from pytorch_lightning import LightningDataModule
from dpr_scale.datamodule.dpr import TRECDataset

class CrossEncoderRerankDataModule(LightningDataModule):
    """
    Parent class for data modules.
    """

    def __init__(self, 
                transform,
                test_path: str,
                test_question_path: str,
                test_passage_path: str,
                test_batch_size: int = 128,  # defaults to val_batch_size
                num_workers: int = 0,  # increasing this bugs out right now
                use_title: bool = False,  # use the title for context passages
                sep_token: str = " [SEP] ",  # sep token between title and passage
                *args,
                **kwargs,):
        super().__init__()
        self.text_transform = transform
        self.test_batch_size = test_batch_size
        self.use_title = use_title
        self.sep_token = sep_token
        self.num_workers = num_workers

        self.datasets = {
            "test": TRECDataset(test_path, test_question_path, test_passage_path)
        }


    def _transform(self, questions, ctxs):
        result = self.text_transform(questions, ctxs)
        return result
    
    def collate(self, batch, stage):
        questions = [row['question'] for row in batch]
        ctxs = [
                maybe_add_title(
                    row["text"], row["title"], self.use_title, self.sep_token
                )
                for row in batch
                ]
        text_tensors = self._transform(questions, ctxs)

        qids = [row["qid"] for row in batch]
        ctx_ids = [row["ctx_id"] for row in batch]
        return {"qid":qids, "ctx_id":ctx_ids, "text_ids": text_tensors}

    def val_dataloader(self):
        return self.test_dataloader()

    def train_dataloader(self):
        return self.test_dataloader()

    def test_dataloader(self):
        sampler = None
        if (
            self.trainer
            and hasattr(self.trainer, "world_size")
            and self.trainer.world_size > 1
        ):
            sampler = ContiguousDistributedSamplerForTest(self.datasets["test"])

        return torch.utils.data.DataLoader(
            self.datasets["test"],
            shuffle=False,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_test,
            sampler=sampler,
        )
    
    def collate_eval(self, batch):
        return self.collate(batch, "eval")

    def collate_test(self, batch):
        return self.collate(batch, "test")

    def collate_train(self, batch):
        return self.collate(batch, "train")
