#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import unittest
import tempfile
from dpr_scale.datamodule.dpr import MemoryMappedDataset, CSVDataset, DenseRetrieverJsonlDataModule, DenseRetrieverPassagesDataModule
from dpr_scale.transforms.hf_bert import BertTransform

# @manual=//python/wheel/transformers3:transformers3
from transformers import BertModel, BertConfig, BertTokenizer
import shutil

def create_bert_tiny(model_dir, vocab_file):
    config = BertConfig(vocab_size=32, hidden_size=16, num_hidden_layers=2, num_attention_heads=1, intermediate_size=4)
    model = BertModel(config)
    model.save_pretrained(model_dir)
    tokenizer = BertTokenizer(vocab_file)
    tokenizer.save_vocabulary(model_dir)

class TestMemoryMappedDataset(unittest.TestCase):
    def setUp(self):
        pwd = os.path.dirname(__file__)
        self.file = os.path.join(pwd, "data/dpr10.jsonl")
        self.input = []
        with open(self.file, "r") as fin:
            for line in fin:
                self.input.append(line)

    def check_dataset_line_by_line(self, dataset):
        # Ensure the dataset reports the correct size.
        self.assertEqual(len(dataset), len(self.input))

        # Ensure the dataset returns the correct data, in the correct order.
        for i in range(len(dataset)):
            self.assertEqual(dataset[i].decode("utf-8"), self.input[i])

    def test_memory_mapped_dataset(self):
        dataset = MemoryMappedDataset(path=self.file, header=False)
        self.check_dataset_line_by_line(dataset)

class TestCSVDataset(unittest.TestCase):
    def setUp(self):
        pwd = os.path.dirname(__file__)
        self.file = os.path.join(pwd, "data/dpr10.tsv")
        self.input = []
        with open(self.file, "r") as fin:
            for line in fin:
                self.input.append(line)

    def test_csv_dataset(self):
        dataset = CSVDataset(path=self.file, sep="\t")
        # Ensure dataset returns the correct data, the first line is the header.
        self.assertEqual(dataset.columns, ["id", "text", "title"])
        self.assertEqual(dataset[0], {"id": "1", "text": "This is a test.", "title": "Test"})
        self.assertEqual(len(dataset)+1, len(self.input))
        for i in range(len(dataset)):
            self.assertEqual(len(dataset.columns), len(dataset[i]))


class TestDenseRetrieverJsonlDataModule(unittest.TestCase):
    def setUp(self):
        pwd = os.path.dirname(__file__)
        self.file = os.path.join(pwd, "data/dpr10.jsonl")
        self.input = []
        with open(self.file, "r") as fin:
            for line in fin:
                self.input.append(line)

        # manually init a tiny transformer and save it to a temp dir
        self.model_dir = tempfile.mkdtemp()
        create_bert_tiny(self.model_dir, vocab_file=os.path.join(pwd, "data/vocab.txt"))

        self.transform = BertTransform(
            model_path=self.model_dir
        )

    def tearDown(self):
        # deleted the temp dir for created tiny transformer
        shutil.rmtree(self.model_dir, ignore_errors=True)


    def check_dataloader(self, dataloader, bsz):
        datapoints = 0
        for batch_idx, batch in enumerate(dataloader):
            # Check if input has correct size
            datapoints += batch["query_ids"]["input_ids"].size(0)
            # For all but the last batch (smaller batches are not discarded) the batchsize should be consistent,
            # for the last batch the size can be less or equal to the defined batchsize
            if batch_idx < len(dataloader) - 1:
                self.assertEqual(batch["pos_ctx_indices"].size(0), bsz)
                self.assertEqual(batch["query_ids"]["input_ids"].size(0), bsz)
            else:
                self.assertLessEqual(batch["pos_ctx_indices"].size(0), bsz)
                self.assertLessEqual(batch["query_ids"]["input_ids"].size(0), bsz)
            self.assertEqual(batch["contexts_ids"]["input_ids"].size(0), batch["ctx_mask"].size(0))
        # Check for overall number of datapoints being correct
        self.assertEqual(datapoints, len(self.input))

    def test_dense_retriever_jsonl_datamodule(self):
        # Test at various batch sizes
        for bsz in [1, 3, 5, 7, 10]:
            dataloader = DenseRetrieverJsonlDataModule(
                transform=self.transform,
                # Dataset args
                train_path= self.file,
                val_path = self.file,
                test_path = self.file,
                batch_size = bsz,
                val_batch_size = bsz,
                test_batch_size = bsz,
                num_positive = 1,
                num_negative = 2,
                neg_ctx_sample = True,
                pos_ctx_sample  = False,
                num_val_negative  = 2,
                num_test_negative  = 50,
                drop_last = False,
                num_workers = 0,
                use_title = True,
                sep_token = " ",
                use_cross_attention = False,
            )
            self.check_dataloader(dataloader.train_dataloader(), bsz)
            self.check_dataloader(dataloader.val_dataloader(), bsz)
            self.check_dataloader(dataloader.test_dataloader(), bsz)


class TestDenseRetrieverPassagesDataModule(unittest.TestCase):
    def setUp(self):
        pwd = os.path.dirname(__file__)
        self.file = os.path.join(pwd, "data/dpr10.tsv")
        self.input = []
        with open(self.file, "r") as fin:
            for line in fin:
                self.input.append(line)

        # manually init a tiny transformer and save it to a temp dir
        self.model_dir = tempfile.mkdtemp()
        create_bert_tiny(self.model_dir, vocab_file=os.path.join(pwd, "data/vocab.txt"))

        self.transform = BertTransform(
            model_path=self.model_dir
        )

    def tearDown(self):
        # deleted the temp dir for created tiny transformer
        shutil.rmtree(self.model_dir, ignore_errors=True)

    def check_dataloader(self, dataloader, bsz):
        datapoints = 0
        for batch_idx, batch in enumerate(dataloader):
            # Check if input has correct size
            datapoints += batch["contexts_ids"]["input_ids"].size(0)
            # For all but the last batch (smaller batches are not discarded) the batchsize should be consistent,
            # for the last batch the size can be less or equal to the defined batchsize
            if batch_idx < len(dataloader) - 1:
                self.assertEqual(batch["contexts_ids"]["input_ids"].size(0), bsz)
            else:
                self.assertLessEqual(batch["contexts_ids"]["input_ids"].size(0), bsz)
        # Check for overall number of datapoints being correct, first line is header
        self.assertEqual(datapoints, len(self.input)-1)

    def test_dense_retriever_passages_datamodule(self):
        # Test at various batch sizes
        for bsz in [1, 3, 5, 7, 10]:
            dataloader = DenseRetrieverPassagesDataModule(
                transform=self.transform,
                test_path = self.file,
                test_batch_size = bsz,
                num_workers = 0,
                use_title = False,
                sep_token = " [SEP] ",
            )
            self.check_dataloader(dataloader.train_dataloader(), bsz)
            self.check_dataloader(dataloader.val_dataloader(), bsz)
            self.check_dataloader(dataloader.test_dataloader(), bsz)
