# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import hydra
import glob
import json
import os
import pickle
import collections
import jsonlines
import torch
from dpr_scale.conf.config import MainConfig
from omegaconf import open_dict
from pytorch_lightning.trainer import Trainer
from dpr_scale.datamodule.dpr import QueryCSVDataset
from dpr_scale.datamodule.citadel import QueryTRECDataset, IDCSVDataset
from typing import Dict, List

def merge_results(
    passages: Dict,
    questions: List,
    top_ids_scores: Dict,
    qrels: Dict = None,
    negative_depth: int = 30,
    dataset="msmarco_passages",
):
    # join passages text with the result ids, their questions
    merged_data = []
    for i, question in zip(range(len(questions)), questions):
        doc_ids = top_ids_scores[question.get("id", i)].get("indexes", [])
        scores = top_ids_scores[question.get("id", i)].get("scores", [])
        if qrels is not None:
            postive_ctxs = []
            negative_ctxs = []
            for id, score in zip(doc_ids, scores):
                if str(id) in qrels[question.get("id", i)]:
                    postive_ctxs.append({
                            "passage_id": passages[id]["id"],
                            "title": passages[id]["title"],
                            "text": passages[id]["text"],
                            "score": float(score),
                            "title_score": 1
                        })
                else:
                    negative_ctxs.append({
                            "passage_id": passages[id]["id"],
                            "title": passages[id]["title"],
                            "text": passages[id]["text"],
                            "score": float(score),
                            "title_score": 1
                        })
                        
            if len(postive_ctxs) == 0:
                continue

            merged_data.append(
                {   "dataset":dataset,
                    "question": question["question"],
                    "question_id": question.get("id", i),
                    "answers": question["answers"] if "answers" in question else [],
                    "positive_ctxs": postive_ctxs,
                    "hard_negative_ctxs": negative_ctxs,
                }
            )
        else:
            ctxs = [
                {
                    "id": passages[id]["id"],
                    "title": passages[id]["title"],
                    "text": passages[id]["text"],
                    "score": float(score),
                }
                for id, score in zip(doc_ids, scores)
            ]

            merged_data.append(
                {
                    "question": question["question"],
                    "answers": question["answers"] if "answers" in question else [],
                    "ctxs": ctxs,
                    "id": question.get("id", i),
                }
            )
    return merged_data

def load_results(paths, tensor=False):
    results = []
    for fname in paths:
        with open(fname, 'rb') as f:
            res = pickle.load(f)
            print(f"Adding {len(res)} results from {fname}")
            results.append(res)
    if tensor:
        results = torch.cat(results, 0)
    else:
        results = sum(results, [])
    return results

def load_qrels(path):
    with open(path) as f:
        lines = f.readlines()
    data = collections.defaultdict(list)
    for line in lines:
        qid, _, doc_id, _ = line.strip().split("\t")
        data[qid].append(doc_id)
    return data

@hydra.main(config_path="../conf", config_name="config")
def main(cfg: MainConfig):
    # Temp patch for datamodule refactoring
    cfg.task.datamodule = None
    if cfg.cross_encoder:
        cfg.datamodule._target_ = (
            "dpr_scale.datamodule.cross_encoder.CrossEncoderRerankDataModule"  # hack
        )   
    else:
        cfg.datamodule._target_ = (
            "dpr_scale.datamodule.citadel.DenseRetrieverRerankDataModule"  # hack
        )

    # trainer.fit does some setup, so we need to call it even though no training is done
    with open_dict(cfg):
        cfg.trainer.limit_train_batches = 0
        if "plugins" in cfg.trainer:
            cfg.trainer.pop(
                "plugins"
            )  # remove ddp_sharded, because it breaks during loading

    print(cfg)
    task = hydra.utils.instantiate(cfg.task, _recursive_=False)
    transform = hydra.utils.instantiate(cfg.task.transform)
    datamodule = hydra.utils.instantiate(cfg.datamodule, transform=transform)
    
    print(f"Writing output to {cfg.task.output_dir}")
    os.makedirs(os.path.dirname(cfg.task.output_dir), exist_ok=True)
    
    trainer = Trainer(**cfg.trainer)
    trainer.test(task, datamodule=datamodule)

    # load questions file
    print(f"Loading questions file {cfg.datamodule.test_question_path}")
    if cfg.datamodule.query_trec:
        questions = QueryTRECDataset(cfg.datamodule.test_question_path, use_id=False)
    else:
        questions = QueryCSVDataset(cfg.datamodule.test_question_path)
    # load all passages:
    print(f"Loading passages from {cfg.datamodule.test_passage_path}")
    ctxs = IDCSVDataset(cfg.datamodule.test_passage_path, use_id=True)

    # reload scores
    mappings = collections.defaultdict(list)
    print("Loading reranking scores.")
    input_paths = sorted(glob.glob(os.path.join(cfg.task.output_dir, "scores_*")))
    scores = load_results(input_paths, tensor=True)
    print("Loading reranking ids.")
    input_paths = sorted(glob.glob(os.path.join(cfg.task.output_dir, "qids_*")))
    qids = load_results(input_paths)
    input_paths = sorted(glob.glob(os.path.join(cfg.task.output_dir, "ctx_ids_*")))
    ctx_ids = load_results(input_paths)

    for qid, ctx_id, score in zip(qids, ctx_ids, scores):
        mappings[qid].append([ctx_id, score.item()])

    top_ids_scores = collections.defaultdict(dict)
    for qid, id_scores in mappings.items():
        ctx_ids, scores = list(zip(*sorted(id_scores, key=lambda x: -x[1])))
        top_ids_scores[qid]["scores"] = scores[:cfg.topk]
        top_ids_scores[qid]["indexes"] = ctx_ids[:cfg.topk]
    qrels = None
    if os.path.exists(cfg.qrel_path) and cfg.create_train_dataset:
        qrels = load_qrels(cfg.qrel_path)
    
    print("Merging results...")
    results = merge_results(ctxs, questions, top_ids_scores, qrels, cfg.dataset)
    if cfg.create_train_dataset:
        with jsonlines.open(os.path.join(cfg.task.output_dir, f"{cfg.dataset}_exp_train_hn.jsonl"), 'w') as g:
            g.write_all(results)
    else:
        with open(os.path.join(cfg.task.output_dir, "rerank.json"), "w") as g:
            g.write(json.dumps(results, indent=4))
            g.write("\n")

if __name__ == "__main__":
    main()
