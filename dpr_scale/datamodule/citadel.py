#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import mmap
import torch
from dpr_scale.utils.utils import (
    ContiguousDistributedSamplerForTest,
    PathManager,
    maybe_add_title,
)
from dpr_scale.datamodule.dpr import (
    MemoryMappedDataset,
    QueryCSVDataset,
    DenseRetrieverDataModuleBase
)

class IDMemoryMappedDataset(MemoryMappedDataset):
    """
    A memory mapped dataset.
    """

    def __init__(self, path, header=False, use_id=False):
        local_path = PathManager.get_local_path(path)
        self.file = open(local_path, mode="r")
        self.mm = mmap.mmap(self.file.fileno(), 0, prot=mmap.PROT_READ)
        self.offset_dict = {}
        if header:
            line = self.mm.readline()
        self.count = 0
        if not use_id:
            self.offset_dict = {0: self.mm.tell()}
            line = self.mm.readline()
            while line:
                self.count += 1
                offset = self.mm.tell()
                self.offset_dict[self.count] = offset
                line = self.mm.readline()

class IDCSVDataset(IDMemoryMappedDataset):
    """
    A memory mapped dataset for csv files
    """

    def __init__(self, path, sep="\t", use_id=False):
        super().__init__(path, header=True, use_id=use_id)
        self.sep = sep
        self.columns = self._get_header()
        if use_id:
            offset = self.mm.tell()
            line = self.mm.readline()
            while line:
                self.count += 1
                pid = self.process_line(line)["id"]
                self.offset_dict[pid] = offset
                offset = self.mm.tell()
                line = self.mm.readline()

    def _get_header(self):
        self.mm.seek(0)
        return self._parse_line(self.mm.readline())

    def _parse_line(self, line):
        """Implementation of csv quoting."""
        row = line.decode().rstrip("\r\n").split(self.sep)
        for i, val in enumerate(row):
            if val and val[0] == '"' and val[-1] == '"':
                row[i] = val.strip('"').replace('""', '"')
        return row

    def process_line(self, line):
        vals = self._parse_line(line)
        if len(self.columns) == len(vals):
            return dict(zip(self.columns, vals))
        else:  # hack
            self.__getitem__(0)



class QueryTRECDataset(IDMemoryMappedDataset):
    """
    A memory mapped dataset for query trec files (such as the test set)
    """

    def __init__(self, path, sep="\t", use_id=False):
        super().__init__(path, header=False, use_id=use_id)
        self.sep = sep
        if use_id:
            offset = self.mm.tell()
            line = self.mm.readline()
            while line:
                self.count += 1
                qid = self.process_line(line)["id"]
                self.offset_dict[qid] = offset
                offset = self.mm.tell()
                line = self.mm.readline()
                
    def _parse_line(self, line):
        """Implementation of csv quoting."""
        row = line.decode().rstrip("\r\n").split(self.sep)
        for i, val in enumerate(row):
            if val and val[0] == '"' and val[-1] == '"':
                row[i] = val.strip('"').replace('""', '"')
        return row

    def process_line(self, line):
        vals = self._parse_line(line)
        return {
            "id": vals[0],
            "question": vals[1],
        }


class TRECDataset(IDMemoryMappedDataset):
    def __init__(self, path, question_path, passage_path, query_trec=True, sep=" "):
        super().__init__(path, header=False)
        self.sep = sep
        self.query_trec = query_trec
        ## read questions and passages as well
        if query_trec:
            self.question_dataset = QueryTRECDataset(question_path, use_id=True)
        else:
            self.question_dataset = QueryCSVDataset(question_path)
        self.passage_dataset = IDCSVDataset(passage_path, use_id=True)
    
    def _parse_line(self, line):
        return line.decode().rstrip("\r\n").split(self.sep)
    
    def process_line(self, line):
        vals = self._parse_line(line)
        qid, ctx_id = vals[0], vals[2]
        if not self.query_trec:
            qid = int(qid)
        question = self.question_dataset[qid]
        passage = self.passage_dataset[ctx_id]
        return {"qid":qid, "ctx_id":ctx_id, "question": question["question"], "text": passage["text"], "title": passage["title"]}


class DenseRetrieverQueriesDataModule(DenseRetrieverDataModuleBase):
    """
    This reads a csv file of questions for query embedding creation.
    """

    def __init__(
        self,
        transform,
        test_path: str,
        test_batch_size: int = 128,  # defaults to val_batch_size
        num_workers: int = 0,  # increasing this bugs out right now
        trec_format: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(transform)
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers

        self.datasets = {
            "test": QueryTRECDataset(test_path) if trec_format else QueryCSVDataset(test_path),
        }
    def collate(self, batch, stage):
        ctx_tensors = self._transform(
            [row['question'] for row in batch]
        )
        inputs = {"query_ids": ctx_tensors, "question": [row['question'] for row in batch]}
        if "id" in batch[0]:
            qids = [row["id"] for row in batch]
            inputs["topic_ids"] = qids
        
        if "answers" in batch[0]:
            answers = [row['answers'] for row in batch]
            inputs["answers"] = answers
        return inputs

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


class DenseRetrieverRerankDataModule(DenseRetrieverDataModuleBase):
    """
    This reads a trec file, question csv file, and passage csv file.
    """

    def __init__(
        self,
        transform,
        test_path: str,
        test_question_path: str,
        test_passage_path: str,
        test_batch_size: int = 128,  # defaults to val_batch_size
        num_workers: int = 0,  # increasing this bugs out right now
        use_title: bool = False,  # use the title for context passages
        sep_token: str = " [SEP] ",  # sep token between title and passage
        query_trec: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(transform)
        self.test_batch_size = test_batch_size
        self.use_title = use_title
        self.sep_token = sep_token
        self.num_workers = num_workers

        self.datasets = {
            "test": TRECDataset(test_path, test_question_path, test_passage_path, query_trec)
        }

    def collate(self, batch, stage):
        question_tensors = self._transform(
            [row['question'] for row in batch]
        )
        ctx_tensors = self._transform(
            [
                maybe_add_title(
                    row["text"], row["title"], self.use_title, self.sep_token
                )
                for row in batch
            ]
        )
        qids = [row["qid"] for row in batch]
        ctx_ids = [row["ctx_id"] for row in batch]
        return {"qid":qids, "ctx_id":ctx_ids, "query_ids": question_tensors, "contexts_ids": ctx_tensors}

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
