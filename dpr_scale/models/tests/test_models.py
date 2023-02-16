#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import unittest
import tempfile
from dpr_scale.models.hf_bert import BertEncoder
from dpr_scale.models.hf_model import HFEncoder
import torch
from torch import nn

# @manual=//python/wheel/transformers3:transformers3
from transformers import BertModel, BertConfig, BertTokenizer
import shutil

def create_bert_tiny(model_dir, vocab_file):
    config = BertConfig(vocab_size=32, hidden_size=16, num_hidden_layers=2, num_attention_heads=1, intermediate_size=4)
    model = BertModel(config)
    model.save_pretrained(model_dir)
    tokenizer = BertTokenizer(vocab_file)
    tokenizer.save_vocabulary(model_dir)

class TestBertEncoder(unittest.TestCase):
    def setUp(self):
        pwd = os.path.dirname(__file__)
        # manually init a tiny transformer and save it to a temp dir
        self.model_dir = tempfile.mkdtemp()
        create_bert_tiny(self.model_dir, vocab_file=os.path.join(pwd, "data/vocab.txt"))

    def tearDown(self):
        # deleted the temp dir for created tiny transformer
        shutil.rmtree(self.model_dir, ignore_errors=True)


    def test_load_bert_encoder(self):
        # with temporarily created model path
        encoder = BertEncoder(
            model_path=self.model_dir,
            dropout=0.1
        )
        self.assertTrue(isinstance(encoder, nn.Module))

        # batch size = 2
        model_inputs = {
            "input_ids": torch.tensor(
                [
                    [2, 1, 3],
                    [4, 5, 2]
                ]
            ),
        }

        sentence_rep = encoder(model_inputs) # B x C
        self.assertEqual(sentence_rep.size(0), 2)
        self.assertEqual(sentence_rep.size(1), 16)

        # batch size = 1
        model_inputs = {
            "input_ids": torch.tensor(
                [
                    [2, 1, 3],
                ]
            ),
        }
        sentence_rep = encoder(model_inputs) # B x C
        self.assertEqual(sentence_rep.size(0), 1)
        self.assertEqual(sentence_rep.size(1), 16)


class TestHFEncoder(unittest.TestCase):
    def setUp(self):
        pwd = os.path.dirname(__file__)
        # manually init a tiny transformer and save it to a temp dir
        self.model_dir = tempfile.mkdtemp()
        create_bert_tiny(self.model_dir, vocab_file=os.path.join(pwd, "data/vocab.txt"))

    def tearDown(self):
        # deleted the temp dir for created tiny transformer
        shutil.rmtree(self.model_dir, ignore_errors=True)


    def test_load_hf_encoder(self):
        # with temporarily created model path
        encoder = HFEncoder(
            model_path=self.model_dir,
            dropout=0.1
        )
        self.assertTrue(isinstance(encoder, nn.Module))

        # batch size = 2
        model_inputs = {
            "input_ids": torch.tensor(
                [
                    [2, 1, 3],
                    [4, 5, 2]
                ]
            ),
        }

        sentence_rep = encoder(model_inputs) # B x C
        self.assertEqual(sentence_rep.size(0), 2)
        self.assertEqual(sentence_rep.size(1), 16)

        # batch size = 1
        model_inputs = {
            "input_ids": torch.tensor(
                [
                    [2, 1, 3],
                ]
            ),
        }
        sentence_rep = encoder(model_inputs) # B x C
        self.assertEqual(sentence_rep.size(0), 1)
        self.assertEqual(sentence_rep.size(1), 16)
