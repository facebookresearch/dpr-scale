#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import pathlib
import pickle
import torch
from dpr_scale.utils.utils import PathManager
from pytorch_lightning.utilities.cloud_io import load as pl_load

import collections
from tqdm import tqdm
import concurrent.futures
from dpr_scale.task.citadel_task import MultiVecRetrieverTask

class GenerateMultiVecEmbeddingsTask(MultiVecRetrieverTask):
    def __init__(self, ctx_embeddings_dir, checkpoint_path, add_context_id, weight_threshold=0., **kwargs):
        super().__init__(**kwargs)
        self.ctx_embeddings_dir = ctx_embeddings_dir
        self.checkpoint_path = checkpoint_path
        self.add_context_id = add_context_id # for token/expert distribution analysis
        self.weight_threshold = weight_threshold # for on-the-fly pruning
        pathlib.Path(ctx_embeddings_dir).mkdir(parents=True, exist_ok=True)

    def setup(self, stage: str):
        super().setup("train")
        print(f"Loading checkpoint from {self.checkpoint_path}")
        checkpoint = pl_load(
            self.checkpoint_path, map_location=lambda storage, loc: storage)
        self.load_state_dict(checkpoint['state_dict'])

    def forward(self, contexts_ids):
        # encode contexts
        contexts_repr = self.encode_contexts(contexts_ids)  # ctx_cnt x d
        return contexts_repr
    
    def encode_contexts(self, contexts_ids):
        contexts_repr = self._encode_sequence(
            contexts_ids, self.context_encoder,**self.context_kwargs
        )  # ctx_cnt x d
        return contexts_repr

    def _eval_step(self, batch, batch_idx):
        contexts_ids = batch["contexts_ids"]  # bs x ctx_cnt x ctx_len
        contexts_repr = self(contexts_ids)
        contexts_repr = {k:v.detach().cpu() for k, v in contexts_repr.items()}
        batch_results = []
        batch_cls = []
        if "cls_repr" in contexts_repr:
            batch_cls = contexts_repr["cls_repr"]
        for batch_id, corpus_id in enumerate(batch["corpus_ids"]):
            results = collections.defaultdict(list)
            for expert_repr, expert_topk_ids, expert_topk_weights, attention_score, context_id in zip(contexts_repr["expert_repr"][batch_id],
                                                                                                contexts_repr["expert_ids"][batch_id],
                                                                                                contexts_repr["expert_weights"][batch_id],
                                                                                                contexts_repr["attention_mask"][batch_id],
                                                                                                contexts_ids["input_ids"].cpu()[batch_id][1:]):
                if attention_score > 0:
                    if len(contexts_repr["expert_ids"].shape) == 2: # COIL and ColBERT
                        if expert_topk_weights > 0:
                            results[expert_topk_ids.item()].append([int(corpus_id), expert_topk_weights, expert_topk_weights * expert_repr])
                    else: # CITADEL
                        for expert_id, expert_weight in zip(expert_topk_ids, expert_topk_weights):
                            if self.add_context_id:
                                results[expert_id.item()].append([int(corpus_id), expert_weight, context_id])
                            else:
                                if expert_weight > self.weight_threshold:
                                    results[expert_id.item()].append([int(corpus_id), expert_weight, expert_weight * expert_repr])
            batch_results.append(results)
        return batch_results, batch_cls

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx)

    def test_epoch_end(self, contexts_reprs):
        def save_file(entry):
            path, output = entry
            with open(path, "wb") as f:
                pickle.dump(output, f, protocol=4)
                
        def parallel_write(expert_dict, output_dir):
            os.makedirs(output_dir, exist_ok=True)
            results = []
            for k, output in expert_dict.items():
                ids, weights, reprs = zip(*output)
                ids = torch.LongTensor(ids)
                weights = torch.stack(weights, 0).to(torch.float32)
                reprs = torch.stack(reprs, 0).to(torch.float32)
                results.append((os.path.join(output_dir, f"{k}.pkl"), (ids, weights, reprs)))
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=1000) as executor:
                for path, output in results:
                    executor.submit(save_file, (path, output))

        expert_embeddings = collections.defaultdict(list)
        cls_embeddings = []
        for batch_contexts_repr, batch_cls in tqdm(contexts_reprs):
            if len(batch_cls) > 0:
                cls_embeddings.append(batch_cls)
            for contexts_repr in batch_contexts_repr:
                for expert_id, res in contexts_repr.items():
                    expert_embeddings[expert_id].extend(res)
        
        if not self.ctx_embeddings_dir:
            self.ctx_embeddings_dir = self.trainer.weights_save_path
            
        if len(cls_embeddings) > 0:
            cls_embeddings = torch.cat(cls_embeddings, 0).to(torch.float32)
            cls_out_path = os.path.join(
                self.ctx_embeddings_dir, f"cls_{self.global_rank:04}.pkl")
            print(f"\nWriting tensors to {cls_out_path}")
            save_file((cls_out_path, cls_embeddings))

        embedding_out_dir = os.path.join(
            self.ctx_embeddings_dir, f"expert_{self.global_rank:04}")
        print(f"\nWriting tensors to {embedding_out_dir}")
        parallel_write(expert_embeddings, embedding_out_dir)
        torch.distributed.barrier()  # make sure rank 0 waits for all to complete

class GenerateMultiVecQueryEmbeddingsTask(GenerateMultiVecEmbeddingsTask):
    def __init__(
        self,
        hnsw_index=False,
        output_path="/tmp/results.jsonl",
        query_emb_output_dir=None,
        passages="",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hnsw_index = hnsw_index
        self.output_path = output_path
        self.query_emb_output_dir = query_emb_output_dir

    def forward(self, query_ids):
        # encode questions
        query_repr = self.encode_queries(query_ids)  # q_cnt x d
        return query_repr

    def encode_queries(self, query_ids):
        query_repr = self._encode_sequence(query_ids, self.query_encoder, **self.query_kwargs)  # bs x d
        return query_repr

    def _eval_step(self, batch, batch_idx):
        query_ids = batch["query_ids"]  # bs x ctx_cnt x ctx_len
        topic_ids = batch["topic_ids"] # add question topic id
        queries_repr = self(query_ids)
        queries_repr = {k:v.detach().cpu() for k, v in queries_repr.items()}
        batch_embeddings = []
        batch_cls = []
        batch_weights = []
        if "cls_repr" in queries_repr:
            batch_cls = queries_repr["cls_repr"]
        for batch_id in range(len(batch["topic_ids"])):
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
                                embeddings[expert_id.item()].append((expert_weight * expert_repr).to(torch.float32)) 
                                weights[expert_id.item()].append(expert_weight.to(torch.float32))
            batch_embeddings.append(embeddings)
            batch_weights.append(weights)
        return batch_embeddings, batch_weights, topic_ids, batch_cls

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx)

    def test_epoch_end(self, queries_reprs):
        embeddings = []
        weights = []
        topic_ids = []
        cls_embeddings = []
        for batch_queries_repr, batch_weights, batch_topic_ids, batch_cls in tqdm(queries_reprs):
            if len(batch_cls) > 0:
                cls_embeddings.append(batch_cls)
            embeddings.extend(batch_queries_repr)
            weights.extend(batch_weights)
            topic_ids.extend(batch_topic_ids)
            
        id_out_file = os.path.join(self.query_emb_output_dir, "query_id.pkl")
        pathlib.Path(id_out_file).parent.mkdir(parents=True, exist_ok=True)
        print(f"\nWriting tensors to {id_out_file}")
        with PathManager.open(id_out_file, mode="wb") as f:
            pickle.dump(topic_ids, f, protocol=4)

        embedding_out_file = os.path.join(self.query_emb_output_dir, "query_repr.pkl")
        pathlib.Path(embedding_out_file).parent.mkdir(parents=True, exist_ok=True)
        print(f"\nWriting tensors to {embedding_out_file}")
        with PathManager.open(embedding_out_file, mode="wb") as f:
            pickle.dump(embeddings, f, protocol=4)

        
        weights_out_file = os.path.join(self.query_emb_output_dir, "query_weight.pkl")
        pathlib.Path(weights_out_file).parent.mkdir(parents=True, exist_ok=True)
        print(f"\nWriting tensors to {weights_out_file}")
        with PathManager.open(weights_out_file, mode="wb") as f:
            pickle.dump(weights, f, protocol=4)
        
        if len(cls_embeddings) > 0:
            cls_embeddings = torch.cat(cls_embeddings, 0)
            cls_out_file = os.path.join(self.query_emb_output_dir, "query_cls.pkl")
            pathlib.Path(cls_out_file).parent.mkdir(parents=True, exist_ok=True)
            print(f"\nWriting tensors to {cls_out_file}")
            with PathManager.open(cls_out_file, mode="wb") as f:
                pickle.dump(cls_embeddings, f, protocol=4)

class RerankMultiVecRetrieverTask(MultiVecRetrieverTask):
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
    
    def sim_score(self, q_repr, ctx_repr):
        scores = torch.sum(q_repr * ctx_repr, 1)
        return scores
    
    def expert_sim_score(self, query_repr, context_repr):
        scores = torch.bmm(query_repr["expert_repr"], context_repr["expert_repr"].permute(0, 2, 1))  # B * LQ * LD
        if "expert_ids" in query_repr:
            if len(query_repr["expert_ids"].size()) == 2:
                # See https://github.com/luyug/COIL/blob/6d15679f9ddb8f29c814d19e674333511c45feb3/modeling.py#L173
                exact_match = query_repr["expert_ids"].unsqueeze(2) == context_repr["expert_ids"].unsqueeze(1)  # B * LQ * LD
                exact_match = exact_match.float()
                if "expert_weights" in query_repr:
                    exact_match_weights = query_repr["expert_weights"].unsqueeze(2) * context_repr["expert_weights"].unsqueeze(1)  # B * LQ * LD    
                    exact_match *= exact_match_weights
                scores = scores * exact_match
            else:
                exact_match = query_repr["expert_ids"].unsqueeze(3).unsqueeze(4) == context_repr["expert_ids"].unsqueeze(1).unsqueeze(2)  # B * LQ * KQ * LD * KD
                if "expert_weights" in query_repr:
                    exact_match_weights = query_repr["expert_weights"].unsqueeze(3).unsqueeze(4) * context_repr["expert_weights"].unsqueeze(1).unsqueeze(2)  # B * LQ * KQ * LD * KD
                    exact_match = torch.where(exact_match, exact_match_weights, torch.tensor([0.], dtype=exact_match_weights.dtype, device=exact_match_weights.device))
                else:
                    exact_match = exact_match.to(query_repr["expert_repr"].dtype)
                scores = scores.unsqueeze(-1).unsqueeze(2)
                scores = scores * exact_match
                scores = scores.view(scores.shape[0], scores.shape[1] * scores.shape[2], scores.shape[3] * scores.shape[4])
        if self.query_pool == "sum":
            scores = scores.max(-1).values.sum(1) # B
        elif self.query_pool == "max":
            scores = scores.max(-1).values.max(1).values # B
        else:
            raise NotImplementedError("Invalid query pooling! Available: [max, sum]")
        return scores

    def encode_queries(self, query_ids):
        query_repr = self._encode_sequence(query_ids, self.query_encoder, **self.query_kwargs)  # bs x d
        return query_repr

    def encode_contexts(self, contexts_ids):
        contexts_repr = self._encode_sequence(
            contexts_ids, self.context_encoder, **self.context_kwargs
        )  # ctx_cnt x d
        return contexts_repr

    def _eval_step(self, batch, batch_idx):
        q_ids = batch["query_ids"]  # bs x q_cnt x q_len
        ctx_ids = batch["contexts_ids"]
        q_repr, ctx_repr = self(q_ids, ctx_ids)
        scores = self.expert_sim_score(q_repr, ctx_repr)
        if "cls_repr" in ctx_repr:
            scores += self.sim_score(q_repr["cls_repr"], ctx_repr["cls_repr"])
        return [batch["qid"], batch["ctx_id"], scores.cpu()]

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
