# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# For faster experimentation on different weights when combining two dense models
# We implement the "re-ranking" scheme: Take top 100 from either model, get their scores,
# and re-rank the top 100 with various weights

import argparse
from copy import copy
from glob import glob
import os
import pickle
import tempfile
from tqdm.notebook import tqdm
from typing import Optional
import ujson as json

import numpy as np
import torch

from dpr_scale.eval_dpr import evaluate_retrieval


def read_pred_json_file(path):
    with open(path) as inf:
        data = json.load(inf)
    return data


def load_passage_embeddings(ctx_embeddings_dir):
    # build index
    input_paths = sorted(glob(os.path.join(ctx_embeddings_dir, "reps_*")))

    vectors = []
    for fname in input_paths:
        with open(fname, 'rb') as f:
            vectors.append(pickle.load(f))
    vectors = torch.cat(vectors, dim=0)
    return vectors


# load question and passage vectors
def load_query_embeddings(ctx_embeddings_dir, query_reps_filename):
    # load question embeddings
    with open(
        os.path.join(ctx_embeddings_dir, query_reps_filename), "rb"
    ) as f:
        q_vectors = pickle.load(f)

    return q_vectors


def rerank_two_predictions_with_weights(
    ctx_emb_dir_1: str,
    ctx_emb_dir_2: str,
    pred_filename: str,
    query_reps_filename: str,
    weights: list,
    output_paths: list,
    topk_1=100,
    topk_2=100,
    topk_out=100,
):
    print("loading predictions...")
    data_1 = read_pred_json_file(os.path.join(ctx_emb_dir_1, pred_filename))
    data_2 = read_pred_json_file(os.path.join(ctx_emb_dir_2, pred_filename))

    print("loading query embeddings...")
    query_emb_1 = load_query_embeddings(ctx_emb_dir_1, query_reps_filename)
    query_emb_2 = load_query_embeddings(ctx_emb_dir_2, query_reps_filename)
    assert len(data_1) == len(query_emb_1) == len(data_2) == len(query_emb_2)

    print("loading passage embeddings...")
    passage_emb_1 = load_passage_embeddings(ctx_emb_dir_1)
    passage_emb_2 = load_passage_embeddings(ctx_emb_dir_2)
    assert len(passage_emb_1) == len(passage_emb_2)

    outputs = [[] for _ in output_paths]

    print("performing joint-pool re-ranking...")
    for i, (q1, q2) in enumerate(tqdm(zip(data_1, data_2))):
        assert q1['question'] == q2['question']
        passages = {}
        ctx_ids = set()
        for ctx in q1['ctxs'][:topk_1]:
            ctx_ids.add(ctx['id'])
            passages[ctx['id']] = ctx
        for ctx in q2['ctxs'][:topk_2]:
            ctx_ids.add(ctx['id'])
            passages[ctx['id']] = ctx
        ctx_ids = sorted([int(x)-1 for x in ctx_ids])
        scores_1 = torch.matmul(
            query_emb_1[i], passage_emb_1[ctx_ids].transpose(0, 1)
        )
        scores_2 = torch.matmul(
            query_emb_2[i], passage_emb_2[ctx_ids].transpose(0, 1)
        )
        assert len(scores_1) == len(scores_2) == len(ctx_ids)

        for i, weight in enumerate(weights):
            scores = scores_1 + scores_2 * weight
            scores, idx = torch.sort(scores, descending=True)

            combined_ctxs = []
            for cidx, score in list(zip(idx, scores))[:topk_out]:
                cid = ctx_ids[cidx.item()]
                cid = str(cid + 1)
                score = score.item()
                combined_ctxs.append({
                    "id": cid,
                    "title": passages[cid]['title'],
                    "text": passages[cid]['text'],
                    "score": score,
                })
            q = copy(q1)
            q['ctxs'] = combined_ctxs
            outputs[i].append(q)
    for output, output_path in zip(outputs, output_paths):
        if os.path.dirname(output_path):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as ouf:
            json.dump(output, ouf, indent=4)


def grid_search_weights(
    ctx_emb_dir_1: str,
    ctx_emb_dir_2: str,
    pred_filename: str,
    query_reps_filename: str,
    weights: list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.43, 1.67, 2, 2.5, 3.33, 5.0, 10.0],
    output_dir: Optional[str] = None,
    eval_on_ks: list = [1, 5, 10, 20, 50, 100],  # evaluate on these Ks
    valid_on_k: int = 100,  # use this accuracy to select the best model
    regex: bool = False,  # whether to use regex for eval (TREC dataset)
):
    assert valid_on_k in eval_on_ks, "The validation criterion is not evaluated."

    if not output_dir:
        output_dir = tempfile.TemporaryDirectory()
    os.makedirs(output_dir, exist_ok=True)
    output_paths = []
    for w in weights:
        output_paths.append(os.path.join(output_dir, f"weight{w}_{pred_filename}"))
    assert len(weights) == len(output_paths)

    rerank_two_predictions_with_weights(
        ctx_emb_dir_1=ctx_emb_dir_1,
        ctx_emb_dir_2=ctx_emb_dir_2,
        pred_filename=pred_filename,
        query_reps_filename=query_reps_filename,
        weights=weights,
        output_paths=output_paths,
    )

    best_acc = -1.0
    best_weight = -1.0
    for weight, output_path in zip(weights, output_paths):
        print("Accuracy for weight", weight)
        acc = evaluate_retrieval(output_path, eval_on_ks, regex=regex)
        acc_k = np.mean(acc[valid_on_k])
        if acc_k > best_acc:
            best_acc = acc_k
            best_weight = weight
    print(
        "The best weight is",
        best_weight,
        f"with top-{valid_on_k} accuracy of {best_acc}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_dir_1', type=str, metavar='path',
                        help="Path to embeddings of model 1.")
    parser.add_argument('--emb_dir_2', type=str, metavar='path',
                        help="Path to embeddings of model 2.")
    parser.add_argument('--pred_filename', type=str)
    parser.add_argument('--query_reps_filename', type=str)
    parser.add_argument('--weights', type=float, nargs='+', default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.43, 1.67, 2, 2.5, 3.33, 5.0, 10.0])
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--regex', action='store_true', default=False, help="regex match")
    parser.add_argument('--eval_on_ks', type=int, nargs='+', default=[1, 5, 10, 20, 50, 100], help="topk to evaluate")
    parser.add_argument('--valid_on_k', type=int, default=100, help="topk to evaluate")
    args = parser.parse_args()

    grid_search_weights(
        args.emb_dir_1,
        args.emb_dir_2,
        args.pred_filename,
        args.query_reps_filename,
        args.weights,
        args.output_dir,
        args.eval_on_ks,
        args.valid_on_k,
        args.regex,
    )
