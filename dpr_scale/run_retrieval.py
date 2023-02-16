# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

# @manual=//faiss/python:pyfaiss
import faiss
import hydra
import glob
import json
import os
import pickle
from dpr_scale.conf.config import MainConfig
from dpr_scale.datamodule.dpr import CSVDataset, QueryCSVDataset
from omegaconf import open_dict
from pytorch_lightning.trainer import Trainer
from typing import Dict, List


def merge_results(
    passages: Dict,
    questions: List,
    top_doc_ids: List,
    scores_list: List,
):
    # join passages text with the result ids, their questions
    merged_data = []
    assert len(top_doc_ids) == len(questions) == len(scores_list)
    for i, question, doc_ids, scores in zip(range(len(questions)), questions, top_doc_ids, scores_list):
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


def build_index(paths):
    index = None
    for fname in paths:
        with open(fname, 'rb') as f:
            vector = pickle.load(f)
            if not index:
                index = faiss.IndexFlatIP(vector.size()[1])
            print(f"Adding {vector.size()} vectors from {fname}")
            index.add(vector.numpy())
    return index


@hydra.main(config_path="conf", config_name="config")
def main(cfg: MainConfig):
    # Temp patch for datamodule refactoring
    cfg.task.datamodule = None
    cfg.task._target_ = (
        "dpr_scale.task.dpr_eval_task.GenerateQueryEmbeddingsTask"  # hack
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

    trainer = Trainer(**cfg.trainer)
    trainer.fit(task, datamodule=datamodule)
    trainer.test(task, datamodule=datamodule)

    # index all passages
    input_paths = sorted(glob.glob(os.path.join(cfg.task.ctx_embeddings_dir, "reps_*")))
    index = build_index(input_paths)

    # reload question embeddings
    print("Loading question vectors.")
    with open(
        task.query_emb_output_path, "rb"
    ) as f:
        q_repr = pickle.load(f)

    print("Retrieving results...")
    scores, indexes = index.search(q_repr.numpy(), 100)

    # load questions file
    print(f"Loading questions file {cfg.datamodule.test_path}")
    questions = QueryCSVDataset(cfg.datamodule.test_path)

    # load all passages:
    print(f"Loading passages from {cfg.task.passages}")
    ctxs = CSVDataset(cfg.task.passages)

    # write output file
    print("Merging results...")
    results = merge_results(ctxs, questions, indexes, scores)
    print(f"Writing output to {cfg.task.output_path}")
    os.makedirs(os.path.dirname(cfg.task.output_path), exist_ok=True)
    with open(cfg.task.output_path, "w") as g:
        g.write(json.dumps(results, indent=4))
        g.write("\n")


if __name__ == "__main__":
    main()
