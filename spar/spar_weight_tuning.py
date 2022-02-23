# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import pickle
import tempfile

from copy import copy
from glob import glob
from p_tqdm import p_map
from tqdm import tqdm
from typing import Optional

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
    output_filename: str,
    query_reps_filename: str,
    weights: list,
    output_paths: list,
    topk_1=100,
    topk_2=100,
    topk_out=200,
):
    print("loading predictions...")
    data_1 = read_pred_json_file(os.path.join(ctx_emb_dir_1, output_filename))
    data_2 = read_pred_json_file(os.path.join(ctx_emb_dir_2, output_filename))

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
                    "score_1": scores_1[cidx].item(),
                    "score_2": scores_2[cidx].item(),
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
        output_filename=pred_filename,
        query_reps_filename=query_reps_filename,
        weights=weights,
        output_paths=output_paths,
    )

    # compute in parallel
    accuracies = p_map(
        evaluate_retrieval,
        [op for op in output_paths],
        [eval_on_ks] * len(output_paths),
        [regex] * len(output_paths),
    )
    for acc in accuracies:
        for k in acc:
            acc[k] = np.mean(acc[k])
    assert len(accuracies) == len(output_paths)
    log_filename = os.path.join(output_dir, f"{pred_filename}.log")
    with open(log_filename, 'w') as log_file:
        best_acc = -1.0
        best_weight = -1.0
        for weight, output_path, acc in zip(weights, output_paths, accuracies):
            # acc = evaluate_retrieval(output_path, eval_on_ks, regex=regex)
            print("Accuracy for weight", weight, "is:", acc, file=log_file)
            acc_k = np.mean(acc[valid_on_k])
            acc_all = np.mean([np.mean(acc[k])*k for k in eval_on_ks])
            print(f"Top-{valid_on_k} accuracy for weight", weight, "is", acc_k)
            print(f"Top-{valid_on_k} accuracy for weight", weight, "is", acc_k, file=log_file)
            if acc_k > best_acc:
                best_acc = acc_k
                best_weight = weight
                best_acc_all = acc_all
            elif acc_k > best_acc - 1e-8 and acc_all > best_acc_all:
                best_acc = acc_k
                best_weight = weight
                best_acc_all = acc_all
        print(
            f"The best weight for {pred_filename} is",
            best_weight,
            f"with top-{valid_on_k} accuracy of {best_acc}"
        )
        print(
            f"The best weight for {pred_filename} is",
            best_weight,
            f"with top-{valid_on_k} accuracy of {best_acc}",
            file=log_file
        )


def grid_search_weights_multiset(
    ctx_emb_dir_1: str,
    ctx_emb_dir_2: str,
    output_dir: Optional[str],
    pred_filenames: list,
    query_reps_filenames: list,
    regexes: list,
    weights: list,
    eval_on_ks: list,  # evaluate on these Ks
    valid_on_k: int,  # use this accuracy to select the best model
):
    assert len(pred_filenames) == len(query_reps_filenames) == len(regexes)
    for pred_filename, query_reps_filename, regex in zip(pred_filenames, query_reps_filenames, regexes):
        grid_search_weights(
            ctx_emb_dir_1=ctx_emb_dir_1,
            ctx_emb_dir_2=ctx_emb_dir_2,
            output_dir=output_dir,
            pred_filename=pred_filename,
            query_reps_filename=query_reps_filename,
            weights=weights,
            eval_on_ks=eval_on_ks,
            valid_on_k=valid_on_k,
            regex=regex,
        )


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_1_emb_dir", type=str, required=True)
    parser.add_argument("--model_2_emb_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--pred_filenames", nargs='+',
        default=[
            "nq_dev.json",
            "squad1_dev.json",
            "trivia_dev.json",
            "webq_dev.json",
            "trec_dev.json",
        ],
        help="filenames to the JSON prediction files; one for each dataset"
    )
    parser.add_argument(
        "--query_reps_filenames", nargs='+',
        default=[
            "query_reps_nq_dev.pkl",
            "query_reps_squad1_dev.pkl",
            "query_reps_trivia_dev.pkl",
            "query_reps_webq_dev.pkl",
            "query_reps_trec_dev.pkl",
        ],
        help="filenames to the query embedding files; one for each dataset"
    )
    parser.add_argument(
        "--use_regex", nargs='+',
        default=[False, False, False, False, True],
        help="whether to use regex (only used for the TREC dataset); one for each dataset"
    )
    parser.add_argument(
        "--weights", nargs="+", type=float,
        default=[
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.43, 1.67, 2, 2.5, 3.33, 5.0, 10.0
        ],
        help="concat weight candidates; for some particular combination of models, the range of the weights may need to be updated (e.g. if the best result is obtained on either end of the candidates)."
    )
    parser.add_argument(
        "--eval_on_ks", nargs="+", type=int,
        default=[1, 5, 10, 20, 50, 100],
        help="which K(s) to evaluate on for top-K retrieval accuracy."
    )
    parser.add_argument(
        "--valid_on_k", type=int, default=100,
        help="use top-k accuracy for model selection; k needs to be in eval_on_ks.",
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    assert len(args.pred_filenames) == len(args.query_reps_filenames) == len(args.use_regex)
    grid_search_weights_multiset(
        ctx_emb_dir_1=args.model_1_emb_dir,
        ctx_emb_dir_2=args.model_2_emb_dir,
        output_dir=args.output_dir,
        pred_filenames=args.pred_filenames,
        query_reps_filenames=args.query_reps_filenames,
        regexes=args.use_regex,
        weights=args.weights,
        eval_on_ks=args.eval_on_ks,
        valid_on_k=args.valid_on_k,
    )
