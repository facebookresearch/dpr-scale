#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import unittest
import tempfile
from dpr_scale.transforms.hf_bert import BertTransform
from dpr_scale.transforms.hf_bert import HFTransform
from dpr_scale.transforms.dpr_transform import DPRTransform
import torch


# @manual=//python/wheel/transformers3:transformers3
from transformers import BertModel, BertConfig, BertTokenizer
import shutil

def create_bert_tiny(model_dir, vocab_file):
    config = BertConfig(vocab_size=32, hidden_size=16, num_hidden_layers=2, num_attention_heads=1, intermediate_size=4)
    model = BertModel(config)
    model.save_pretrained(model_dir)
    tokenizer = BertTokenizer(vocab_file)
    tokenizer.save_vocabulary(model_dir)

class TestBertTransform(unittest.TestCase):
    def setUp(self):
        pwd = os.path.dirname(__file__)
        # manually init a tiny transformer and save it to a temp dir
        self.model_dir = tempfile.mkdtemp()
        create_bert_tiny(self.model_dir, vocab_file=os.path.join(pwd, "data/vocab.txt"))

    def tearDown(self):
        # deleted the temp dir for created tiny transformer
        shutil.rmtree(self.model_dir, ignore_errors=True)


    def test_bert_transform(self):
        # with temporarily created model path
        transform = BertTransform(
            model_path=self.model_dir,
            max_seq_len=256,
            add_special_tokens = True,
            return_tensors = True
        )

        outputs = transform(["a b c d"])
        expected = {'input_ids': torch.tensor([[3, 6, 7, 8, 9, 4]])}

        self.assertTrue(isinstance(outputs["input_ids"], torch.Tensor))
        self.assertEqual(outputs['input_ids'].size(1), 6)
        self.assertTrue(torch.all(outputs["input_ids"].eq(expected["input_ids"])))

        outputs = transform(["d e f g a b", "d c b a"])
        expected = {'input_ids': torch.tensor(
            [[ 3,  9, 10, 11, 12,  6,  7,  4], [ 3,  9,  8,  7,  6,  4,  0,  0]])
        }
        self.assertTrue(isinstance(outputs["input_ids"], torch.Tensor))
        self.assertEqual(outputs['input_ids'].size(1), 8)
        self.assertTrue(torch.all(outputs["input_ids"].eq(expected["input_ids"])))



class TestHFTransform(unittest.TestCase):
    def setUp(self):
        pwd = os.path.dirname(__file__)
        # manually init a tiny transformer and save it to a temp dir
        self.model_dir = tempfile.mkdtemp()
        create_bert_tiny(self.model_dir, vocab_file=os.path.join(pwd, "data/vocab.txt"))

    def tearDown(self):
        # deleted the temp dir for created tiny transformer
        shutil.rmtree(self.model_dir, ignore_errors=True)


    def test_hf_transform(self):
        # with temporarily created model path
        transform = HFTransform(
            model_path=self.model_dir,
            max_seq_len=256,
            add_special_tokens = True,
            return_tensors = True
        )

        outputs = transform(["a b c d"])
        expected = {'input_ids': torch.tensor([[3, 6, 7, 8, 9, 4]])}

        self.assertTrue(isinstance(outputs["input_ids"], torch.Tensor))
        self.assertEqual(outputs['input_ids'].size(1), 6)
        self.assertTrue(torch.all(outputs["input_ids"].eq(expected["input_ids"])))

        outputs = transform(["d e f g a b", "d c b a"])
        expected = {'input_ids': torch.tensor(
            [[ 3,  9, 10, 11, 12,  6,  7,  4], [ 3,  9,  8,  7,  6,  4,  0,  0]])
        }
        self.assertTrue(isinstance(outputs["input_ids"], torch.Tensor))
        self.assertEqual(outputs['input_ids'].size(1), 8)
        self.assertTrue(torch.all(outputs["input_ids"].eq(expected["input_ids"])))


class TestDPRTransform(unittest.TestCase):
    def setUp(self):
        pwd = os.path.dirname(__file__)
        # manually init a tiny transformer and save it to a temp dir
        self.model_dir = tempfile.mkdtemp()
        create_bert_tiny(self.model_dir, vocab_file=os.path.join(pwd, "data/vocab.txt"))

        self.data = []
        with open(os.path.join(pwd, "data/dpr10.jsonl"), "r") as fin:
            for line in fin:
                self.data.append(line)

    def tearDown(self):
        # deleted the temp dir for created tiny transformer
        shutil.rmtree(self.model_dir, ignore_errors=True)


    def test_dpr_transform(self):
        # with temporarily created model path
        text_transform = HFTransform(
            model_path=self.model_dir,
            max_seq_len=256,
            add_special_tokens = True,
            return_tensors = True
        )

        dpr_transform = DPRTransform(
            text_transform,
            num_positive= 1,
            num_negative= 7,
            neg_ctx_sample= True,
            pos_ctx_sample= False,
            num_val_negative= 7,
            num_test_negative = None,
            use_title= True,
            sep_token = " "
        )

        dpr_transform._transform(["a b c"])
        dpr_transform._transform(["a b c", "d e f"])

        # test at various batch sizes
        for bsz in [1, 3, 5, 7, 10]:
            outputs = dpr_transform(self.data[:bsz])
            self.assertTrue('query_ids' in outputs)
            self.assertTrue('contexts_ids' in outputs)
            self.assertTrue('pos_ctx_indices' in outputs)
            self.assertTrue('ctx_mask' in outputs)
            self.assertTrue(isinstance(outputs['query_ids']['input_ids'], torch.Tensor))
