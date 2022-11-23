#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import collections
import torch
import time
import json
from tqdm import tqdm
from dpr_scale.task.citadel_task import MultiVecRetrieverTask
from dpr_scale.datamodule.citadel import IDCSVDataset
from dpr_scale.index.inverted_vector_index import IVFGPUIndex, IVFCPUIndex, IVFPQGPUIndex, IVFPQCPUIndex
from pytorch_lightning.utilities.cloud_io import load as pl_load

class CITADELRetrievalTask(MultiVecRetrieverTask):
    def __init__(
        self,
        ctx_embeddings_dir,
        checkpoint_path,
        index2docid_path=None,
        hnsw_index=False,
        output_path="/tmp/results.jsonl",
        passages="",
        topk=100,
        cuda=True,
        portion=1.0,
        quantizer=None,
        sub_vec_dim=4,
        expert_parallel=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ctx_embeddings_dir = ctx_embeddings_dir
        self.checkpoint_path = checkpoint_path
        self.index2docid_path = index2docid_path
        self.hnsw_index = hnsw_index
        self.output_path = output_path
        self.passages = passages
        self.topk = topk
        self.cuda = cuda
        self.quantizer = quantizer if quantizer != "None" else None
        self.sub_vec_dim = sub_vec_dim
        self.portion = portion
        self.expert_parallel = expert_parallel
        self.latency = collections.defaultdict(float)

    def setup(self, stage: str):
        super().setup("train")
        print(f"Loading checkpoint from {self.checkpoint_path}")
        checkpoint = pl_load(
            self.checkpoint_path, map_location=lambda storage, loc: storage)
        self.load_state_dict(checkpoint['state_dict'])
        print(f"Loading passages from {self.passages}")
        self.ctxs = IDCSVDataset(self.passages, use_id=True)
        print("Setting up index...")
        if self.cuda:
            if self.quantizer == "pq":
                self.index = IVFPQGPUIndex(self.portion, len(self.ctxs), self.ctx_embeddings_dir, self.expert_parallel, sub_vec_dim=self.sub_vec_dim)
            else:
                self.index = IVFGPUIndex(self.portion, len(self.ctxs), self.ctx_embeddings_dir, self.expert_parallel)
        else:
            if self.quantizer == "pq":
                self.index = IVFPQCPUIndex(self.portion, len(self.ctxs), self.ctx_embeddings_dir, sub_vec_dim=self.sub_vec_dim)
            else:
                self.index = IVFCPUIndex(self.portion, len(self.ctxs), self.ctx_embeddings_dir)

    def forward(self, query_ids):
        # encode questions
        query_repr = self.encode_queries(query_ids)  # q_cnt x d
        return query_repr

    def encode_queries(self, query_ids):
        query_repr = self._encode_sequence(query_ids, self.query_encoder, **self.query_kwargs)  # bs x d
        return query_repr

    def _eval_step(self, batch, batch_idx):
        tic = time.perf_counter()
        query_ids = batch["query_ids"]  # bs x ctx_cnt x ctx_len
        topic_ids = batch["topic_ids"] if "topic_ids" in batch else [] # add question topic id
        answers = batch["answers"] if "answers" in batch else [] # add question answers
        questions = batch["question"] if "question" in batch else [] # add question

        queries_repr = self(query_ids)
        queries_repr = {k:v.detach() for k, v in queries_repr.items()}

        batch_embeddings = []
        batch_weights = []
        batch_cls = []
        if "cls_repr" in queries_repr:
            batch_cls = queries_repr["cls_repr"]

        iterator = topic_ids if len(topic_ids) > 0 else list(range(len(query_ids["input_ids"]))) 
        for batch_id in range(len(iterator)):
            embeddings = collections.defaultdict(list)
            weights = collections.defaultdict(list)
            for expert_repr, expert_topk_ids, expert_topk_weights, attention_score in zip(queries_repr["expert_repr"][batch_id],
                                                                                        queries_repr["expert_ids"][batch_id],
                                                                                        queries_repr["expert_weights"][batch_id],
                                                                                        queries_repr["attention_mask"][batch_id]):
                if attention_score > 0:
                    if len(queries_repr["expert_ids"].shape) == 2:
                        embeddings[expert_topk_ids.item()].append((expert_topk_weights * expert_repr).to(torch.float32))
                        weights[expert_topk_ids.item()].append(expert_topk_weights.to(torch.float32))
                    else:
                        for expert_id, expert_weight in zip(expert_topk_ids, expert_topk_weights):
                            if expert_weight > 0:
                                embeddings[expert_id.item()].append((expert_weight * expert_repr).to(torch.float16)) 
                                weights[expert_id.item()].append(expert_weight.to(torch.float16))
            batch_embeddings.append(embeddings)
            batch_weights.append(weights)
        toc = time.perf_counter()
        self.latency["encode_time"] += toc - tic
        
        batch_top_scores, batch_top_ids = self.index.search(batch_cls, batch_embeddings, batch_weights, self.topk) 
        if self.cuda:
            return batch_top_scores.cpu().tolist(), batch_top_ids.cpu().tolist(), topic_ids, questions, answers
        else:
            return batch_top_scores.tolist(), batch_top_ids.tolist(), topic_ids, questions, answers

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx)

    def test_epoch_end(self, queries_reprs):
        top_scores = []
        top_ids = []
        topic_ids = []
        questions = []
        answers = []
        for batch_top_scores, batch_top_ids, batch_topic_ids, batch_questions, batch_answers in tqdm(queries_reprs):
            top_scores.extend(batch_top_scores)
            top_ids.extend(batch_top_ids)
            topic_ids.extend(batch_topic_ids)
            questions.extend(batch_questions)
            answers.extend(batch_answers)
        
        self.latency["encode_time"] += self.index.latency["encode_time"]
        self.index.latency.pop("encode_time")
        print(self.latency)
        print(self.index.latency)
        
        print("Merging results...")
        if len(topic_ids) > 0:
            trec_reults = self.merge_trec_results(topic_ids, top_ids, top_scores)
            print(f"Writing output to {self.output_path}")
            os.makedirs(self.output_path, exist_ok=True)
            with open(os.path.join(self.output_path, f"retrieval_{self.global_rank:04}.trec"), "w") as g:
                g.writelines(trec_reults)
        
        elif len(answers) > 0:
            qa_reults = self.merge_qa_results(questions, answers, top_ids, top_scores)
            print(f"Writing output to {self.output_path}")
            os.makedirs(self.output_path, exist_ok=True)
            with open(os.path.join(self.output_path, f"retrieval_{self.global_rank:04}.json"), "w") as g:
                g.write(json.dumps(qa_reults, indent=4))
                g.write("\n")
        torch.distributed.barrier()  # make sure rank 0 waits for all to complete

    def merge_trec_results(
        self,
        topic_ids,
        top_doc_ids,
        scores_list,
    ):
        # join passages text with the result ids, their questions
        i2d = []
        if self.index2docid_path is not None and os.path.exists(self.index2docid_path):
            with open(self.index2docid_path) as f:
                lines = f.readlines()
            for line in lines:
                i2d.append(line.strip())
        
        trec_data = []
        assert len(top_doc_ids) == len(topic_ids) == len(scores_list)
        for topic_id, doc_ids, scores in zip(topic_ids, top_doc_ids, scores_list):
            for rank, (doc_id, score) in enumerate(zip(doc_ids, scores)): 
                if len(i2d) > 0:
                    trec_data.append(f"{topic_id} Q0 {i2d[doc_id]} {rank+1} {score:.6f} dpr-scale\n")
                else:
                    trec_data.append(f"{topic_id} Q0 {doc_id} {rank+1} {score:.6f} dpr-scale\n")
        return trec_data
        
        
    def merge_qa_results(
        self,
        questions,
        answers,
        top_doc_ids,
        scores_list,
    ):
        qa_data = []
        assert len(top_doc_ids) == len(answers) == len(scores_list)
        for question, answer, doc_ids, scores in zip(questions, answers, top_doc_ids, scores_list):
            ctxs = [
                {
                    "id": self.ctxs[str(id)]["id"],
                    "title": self.ctxs[str(id)]["title"],
                    "text": self.ctxs[str(id)]["text"],
                    "score": float(score),
                }
                for id, score in zip(doc_ids, scores)
            ]

            qa_data.append(
                {
                    "question": question,
                    "answers": answer,
                    "ctxs": ctxs,
                }
            )
        return qa_data
