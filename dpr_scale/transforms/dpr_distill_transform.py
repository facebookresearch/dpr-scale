#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import random
from typing import Any, Dict, List

import hydra
import torch
import torch.nn as nn
import ujson  # @manual=third-party//ultrajson:ultrajson


class DPRDistillTransform(nn.Module):
    def __init__(
        self,
        text_transform,
        pos_ctx_sample: bool = True,  # sample from positives list
        text_column: str = "text",  # for onbox
    ):
        super().__init__()
        if isinstance(text_transform, nn.Module):
            self.text_transform = text_transform
        else:
            self.text_transform = hydra.utils.instantiate(text_transform)
        self.pos_ctx_sample = pos_ctx_sample
        self.text_column = text_column

    def _transform(self, texts: List[str]) -> List[List[int]]:
        result = self.text_transform({"text": texts})["token_ids"]
        return result

    def forward(self, batch: Dict[str, Any], stage: str = "train") -> Dict[str, Any]:
        """
        Combines the question and the associated positive context embedding or
        question embeddings. Ensures that each questions has two embeddings
        (i.e.two rows), one is for the question itself embedding, and another is
        for one of sampled positive context embedding. For sampling postive context
        embeddings, we randomly sample single embedding from all positive context embeddings
        associated with the target question.
        """
        all_target_tensor = []
        all_questions = []
        rows = batch if type(batch) is list else batch[self.text_column]

        for row in rows:
            row = ujson.loads(row)

            # retrieve ctx and question vectors
            pos_ctx_vec = row["ctx_target_vectors"]
            question_vec = row["qry_target_vector"]

            # check the pos_ctx_vec type
            assert (
                len(pos_ctx_vec) > 0
            ), f"No Positive Contexts in Row '{row['question']}'."
            assert isinstance(
                pos_ctx_vec[0], list
            ), f"Positive Contexts needs to be a list of embeddings in Row '{row['question']}'."

            # sample only one vector
            if stage == "train" and self.pos_ctx_sample:
                sampled_tensor = random.sample(pos_ctx_vec, 1)
            else:
                # just take the first one
                sampled_tensor = pos_ctx_vec[:1]

            # take questions
            question = row["question"]

            all_questions.extend([question] * 2)
            all_target_tensor.extend(torch.Tensor(sampled_tensor))
            all_target_tensor.append(torch.Tensor(question_vec))

        # tokenize and stack
        question_tensors = self._transform(all_questions)
        target_tensors = torch.stack(all_target_tensor, dim=0)

        return {
            "query_ids": question_tensors,
            "target_vectors": target_tensors,
        }
