#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import hydra
import numpy as np
import torch
import torch.nn as nn
import ujson  # @manual=third-party//ultrajson:ultrajson
from dpr_scale.transforms.hf_transform import HFTransform
from dpr_scale.utils.utils import maybe_add_title


class DPRTransform(nn.Module):
    def __init__(
        self,
        text_transform,
        num_positive: int = 1,  # currently, like the original paper only 1 is supported
        num_negative: int = 7,
        neg_ctx_sample: bool = True,  # sample from negative list
        pos_ctx_sample: bool = False,  # sample from positives list
        num_val_negative: int = 7,  # num negatives to use in validation
        num_test_negative=None,  # defaults to num_val_negative
        use_title: bool = False,  # use the title for context passages
        sep_token: str = " ",  # sep token between title and passage
        rel_sample: bool = False,  # Use relevance scores to sample ctxs
        text_column: str = "text",  # for onbox
    ):
        super().__init__()
        if num_positive > 1:
            raise ValueError(
                "Only 1 positive example is supported. Update the loss to support more!"
            )
        if isinstance(text_transform, nn.Module):
            self.text_transform = text_transform
        else:
            self.text_transform = hydra.utils.instantiate(text_transform)
        self.num_positive = num_positive
        self.num_negative = num_negative
        self.neg_ctx_sample = neg_ctx_sample
        self.pos_ctx_sample = pos_ctx_sample
        self.num_val_negative = num_val_negative
        self.num_test_negative = (
            num_test_negative if num_test_negative else self.num_val_negative
        )
        self.use_title = use_title
        self.sep_token = sep_token
        if isinstance(self.text_transform, HFTransform):
            self.sep_token = self.text_transform.sep_token
        self.text_column = text_column
        self.rel_sample = rel_sample

    def _transform(self, texts):
        if not isinstance(self.text_transform, HFTransform):
            result = self.text_transform({"text": texts})["token_ids"]
        else:
            result = self.text_transform(texts)
        return result

    def forward(self, batch, stage="train"):
        """
        Combines pos and neg contexts. Samples randomly limited number of pos/neg
        contexts if stage == "train". Also ensures we have exactly num_negative
        contexts by padding with fake contexts if the data did not have enough.
        A boolean mask is created to ignore these fake contexts when training.
        """
        questions = []
        all_ctxs = []
        positive_ctx_indices = []
        ctx_mask = []
        scores = []
        rows = batch if type(batch) is list else batch[self.text_column]
        for row in rows:
            row = ujson.loads(row)
            # also support DPR output format
            if "positive_ctxs" not in row and "ctxs" in row:
                row["positive_ctxs"] = []
                row["hard_negative_ctxs"] = []
                for ctx in row["ctxs"]:
                    if ctx["has_answer"]:
                        row["positive_ctxs"].append(ctx)
                    else:
                        row["hard_negative_ctxs"].append(ctx)
                if not row["positive_ctxs"]:
                    row["positive_ctxs"].append(row["ctxs"][0])

            # sample positive contexts
            contexts_pos = row["positive_ctxs"]

            # Handle case when context is a list of tokens instead of string.
            try:
                if len(contexts_pos) > 0:
                    assert isinstance(contexts_pos[0]["text"], str)
            except AssertionError:
                for c in contexts_pos:
                    c["text"] = " ".join(c["text"])

            if stage == "train" and self.pos_ctx_sample:
                rel = [
                    ctx.get("relevance", 1.0) if self.rel_sample else 1.0
                    for ctx in contexts_pos
                ]
                # normalize rel to a probability.
                proba = [float(r) / sum(rel) for r in rel]
                contexts_pos = np.random.choice(
                    contexts_pos, self.num_positive, replace=False, p=proba
                ).tolist()
            else:
                contexts_pos = contexts_pos[: self.num_positive]

            # sample negative contexts
            contexts_neg = row["hard_negative_ctxs"]
            if stage == "train":
                num_neg_sample = self.num_negative
            elif stage == "eval":
                num_neg_sample = self.num_val_negative
            elif stage == "test":
                num_neg_sample = self.num_test_negative

            if num_neg_sample > 0:
                if (
                    stage == "train"
                    and self.neg_ctx_sample
                    and len(contexts_neg) > num_neg_sample
                ):
                    rel = [
                        ctx.get("relevance", 1.0) if self.rel_sample else 1.0
                        for ctx in contexts_neg
                    ]
                    # normalize rel to a probability.
                    proba = [float(r) / sum(rel) for r in rel]
                    contexts_neg = np.random.choice(
                        contexts_neg, num_neg_sample, replace=False, p=proba
                    ).tolist()
                else:
                    contexts_neg = contexts_neg[:num_neg_sample]
            else:
                contexts_neg = []

            ctxs = contexts_pos + contexts_neg
            # pad this up to num_neg_sample contexts
            mask = [0] * len(ctxs)
            if len(contexts_neg) < num_neg_sample:
                # add dummy ctxs
                ctxs.extend(
                    [{"text": "0", "title": "0", "score": 0}]
                    * (num_neg_sample - len(contexts_neg))
                )
                mask.extend([1] * (num_neg_sample - len(contexts_neg)))
            # make sure all rows have same context count
            assert len(ctxs) == (
                self.num_positive + num_neg_sample
            ), f"Row has improper ctx count. Check positive ctxs in: {row}"
            # teacher's retrieval score for distillation
            scores.append([x["score"] if "score" in x else 0 for x in ctxs])
            current_ctxs_len = len(all_ctxs)
            all_ctxs.extend(ctxs)
            positive_ctx_indices.append(current_ctxs_len)
            questions.append(row["question"])
            ctx_mask.extend(mask)

        ctx_text = [
            maybe_add_title(x["text"], x["title"], self.use_title, self.sep_token)
            for x in all_ctxs
        ]
        question_tensors = self._transform(questions)
        ctx_tensors = self._transform(ctx_text)
        return {
            "query_ids": question_tensors,
            "contexts_ids": ctx_tensors,
            "pos_ctx_indices": torch.tensor(positive_ctx_indices, dtype=torch.long),
            "scores": torch.tensor(scores, dtype=torch.float32),
            "ctx_mask": torch.tensor(ctx_mask, dtype=torch.bool),
        }


class DPRCrossAttentionTransform(DPRTransform):
    def __init__(
        self,
        text_transform,
        num_positive: int = 1,
        num_negative: int = 7,
        neg_ctx_sample: bool = True,
        pos_ctx_sample: bool = False,
        num_val_negative: int = 7,  # num negatives to use in validation
        num_test_negative: int = None,  # defaults to num_val_negative
        use_title: bool = False,  # Not supported for now
        sep_token: str = " ",  # sep token between question and passage
        text_column: str = "text",  # for onbox
        num_random_negs: int = 0,
        rel_sample: bool = False,  # Not supported for now
    ):
        super().__init__(
            text_transform,
            num_positive,
            num_negative,
            neg_ctx_sample,
            pos_ctx_sample,
            num_val_negative,
            num_test_negative,
            use_title,
            sep_token,
            rel_sample,
            text_column,
        )
        self.num_random_negs = num_random_negs

    def forward(self, batch, stage="train"):
        """
        Combines pos and neg contexts. Samples randomly limited number of pos/neg
        contexts if stage == "train". Then concatenates them for cross attention
        training along with the labels.
        """
        all_ctxs = []
        all_labels = []
        rows = batch if type(batch) is list else batch[self.text_column]
        neg_candidates = []
        for row in rows:
            # collect candiates for random in-batch negatives
            row = ujson.loads(row)
            neg_candidates.extend(row["positive_ctxs"])
            neg_candidates.extend(row["hard_negative_ctxs"])
        for row in rows:
            row = ujson.loads(row)
            # also support DPR output format
            if "positive_ctxs" not in row and "ctxs" in row:
                row["positive_ctxs"] = []
                row["hard_negative_ctxs"] = []
                for ctx in row["ctxs"]:
                    if ctx["has_answer"]:
                        row["positive_ctxs"].append(ctx)
                    else:
                        row["hard_negative_ctxs"].append(ctx)
                if not row["positive_ctxs"]:
                    row["positive_ctxs"].append(row["ctxs"][0])

            # sample positive contexts
            contexts_pos = row["positive_ctxs"]

            # Handle case when context is a list of tokens instead of string.
            try:
                assert isinstance(contexts_pos[0]["text"], str)
            except AssertionError:
                for c in contexts_pos:
                    c["text"] = " ".join(c["text"])

            if stage == "train" and self.pos_ctx_sample:
                rel = [
                    ctx.get("relevance", 1.0) if self.rel_sample else 1.0
                    for ctx in contexts_pos
                ]
                # normalize rel to a probability.
                proba = [float(r) / sum(rel) for r in rel]
                contexts_neg = np.random.choice(
                    contexts_pos, self.num_positive, replace=False, p=proba
                ).tolist()
            else:
                contexts_pos = contexts_pos[: self.num_positive]

            # sample negative contexts
            contexts_neg = row["hard_negative_ctxs"]
            num_random_negs = 0
            if stage == "train":
                num_neg_sample = self.num_negative
                num_random_negs = self.num_random_negs
            elif stage == "eval":
                num_neg_sample = self.num_val_negative
            elif stage == "test":
                num_neg_sample = self.num_test_negative

            if num_neg_sample > 0:
                if (
                    stage == "train"
                    and self.neg_ctx_sample
                    and len(contexts_neg) > num_neg_sample
                ):
                    rel = [
                        ctx.get("relevance", 1.0) if self.rel_sample else 1.0
                        for ctx in contexts_neg
                    ]
                    # normalize rel to a probability.
                    proba = [float(r) / sum(rel) for r in rel]
                    contexts_neg = np.random.choice(
                        contexts_neg, num_neg_sample, replace=False, p=proba
                    ).tolist()
                else:
                    contexts_neg = contexts_neg[:num_neg_sample]
            else:
                contexts_neg = []
            # Concat texts with sep token
            ctxs = contexts_pos + contexts_neg
            if len(contexts_neg) < num_neg_sample + num_random_negs:
                # add dummy ctxs
                ctxs.extend(
                    np.random.choice(
                        neg_candidates,
                        (num_neg_sample + num_random_negs - len(contexts_neg)),
                        replace=False,
                    ).tolist()
                )
            concat_ctxs = [
                maybe_add_title(ctx["text"], row["question"], True, self.sep_token)
                for ctx in ctxs
            ]
            all_ctxs.extend(concat_ctxs)
            all_labels.append("0")

        return self.text_transform(
            {
                "text": all_ctxs,
                "label": all_labels,
            }
        )
