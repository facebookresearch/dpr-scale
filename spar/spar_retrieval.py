# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from glob import glob
from tqdm import tqdm

import argparse
import gc
import os
import pickle
import json
import csv
import faiss
import torch


def build_index(vectors, batch_size=100000):
    index = faiss.IndexFlatIP(vectors.size()[1])
    for i in range(0, len(vectors), batch_size):
        print(f"adding {i}..{i+batch_size} to index")
        v = vectors[i : i + batch_size]
        index.add(v.numpy())
    return index


def load_test_dataset(jsonl_dataset_path):
    with open(jsonl_dataset_path) as f:
        questions = [json.loads(line) for line in f]
    print(f"Loaded {len(questions)} questions.")
    return questions


def load_passages_tsv(tsv_passages_path):
    passages = []
    with open(tsv_passages_path) as inf:
        reader = csv.reader(inf, delimiter='\t')
        for row in reader:
            if row[0] == 'id':
                headers = {h:i for i, h in enumerate(row)}
                continue
            ctx = {
                'id': row[headers['id']],
                'title': row[headers['title']],
                'text': row[headers['text']],
            }
            passages.append(ctx)
    print(f"Loaded {len(passages)} passages.")
    return passages


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


def dense_search(questions, q_vectors, passages, index, topk):
    all_scores, all_indices = index.search(q_vectors.numpy(), topk)
    assert len(questions) == len(all_scores) == len(all_indices)
    print("Dense search finished.")

    results = []
    for i, question, indices, scores in tqdm(zip(range(len(questions)), questions, all_indices, all_scores)):
        ctxs = []
        for idx, score in zip(indices, scores):
            ctxs.append({
                    "id": passages[idx]["id"],
                    "title": passages[idx]["title"],
                    "text": passages[idx]["text"],
                    "score": float(score),
            })
        results.append({
            "question": question["question"],
            "answers": question["answers"] if "answers" in question else [],
            "ctxs": ctxs,
            "id": question.get("id", str(i)),
        })
    return results


def run_spar_retrieval(
    jsonl_dataset_paths,
    tsv_passages_path,
    ctx_embeddings_dir_1,
    ctx_embeddings_dir_2,
    output_dir,
    output_filenames,
    query_emb_names=['query_reps.pkl'],
    weights=None,
    save_embeddings=False,
    topk=100,
    pooling='concat',
):
    gc.collect()
    assert len(jsonl_dataset_paths) == len(query_emb_names) == len(output_filenames)
    if not weights:
        weights = [1.0] * len(jsonl_dataset_paths)
    assert len(weights) == len(query_emb_names)

    print("loading questions...")
    # load original test dataset
    questions_list = [
        load_test_dataset(jsonl_dataset_path)
        for jsonl_dataset_path in jsonl_dataset_paths
    ]
    print("loading passages...")
    # load passages tsv
    passages = load_passages_tsv(tsv_passages_path)

    print("loading model_1 (base retriever) vectors...")
    p_vectors_1 = load_passage_embeddings(ctx_embeddings_dir_1)
    q_vectors_1_list = [
        load_query_embeddings(ctx_embeddings_dir_1, query_emb_name)
        for query_emb_name in query_emb_names
    ]

    print("loading model_2 (lambda) vectors...")
    p_vectors_2 = load_passage_embeddings(ctx_embeddings_dir_2)
    q_vectors_2_list = [
        load_query_embeddings(ctx_embeddings_dir_2, query_emb_name)
        for query_emb_name in query_emb_names
    ]

    assert len(passages) == len(p_vectors_1) == len(p_vectors_2)
    for q_vectors_1, q_vectors_2 in zip(q_vectors_1_list, q_vectors_2_list):
        assert len(q_vectors_1) == len(q_vectors_2)

    q_vectors_list = []
    # apply weights to the query vectors
    for questions, q_vectors_1, q_vectors_2, weight in zip(
        questions_list, q_vectors_1_list, q_vectors_2_list, weights
    ):
        if pooling.lower() == 'concat':
            print("concat question vectors...")
            q_vectors = torch.cat([q_vectors_1, weight * q_vectors_2], dim=-1)
        elif pooling.lower() == 'mean':
            print("averaging question vectors...")
            q_vectors = (q_vectors_1 + weight * q_vectors_2) / (1.0 + weight)
        elif pooling.lower() == 'sum':
            print("summing question vectors...")
            q_vectors = (q_vectors_1 + weight * q_vectors_2)
        else:
            raise ValueError(pooling)
        assert len(questions) == len(q_vectors)
        q_vectors_list.append(q_vectors)
    assert len(q_vectors_list) == len(questions_list)

    os.makedirs(output_dir, exist_ok=True)
    if save_embeddings:
        for q_vectors, query_emb_output_filename in zip(q_vectors_list, query_emb_names):
            with open(os.path.join(output_dir, query_emb_output_filename), 'wb') as ouf:
                pickle.dump(q_vectors, ouf, protocol=4)

    if pooling.lower() == 'concat':
        print("concat passage vectors...")
        p_vectors = torch.cat([p_vectors_1, p_vectors_2], dim=-1)
    elif pooling.lower() == 'mean':
        print("averaging passage vectors...")
        p_vectors = (p_vectors_1 + p_vectors_2) / 2.0
    elif pooling.lower() == 'sum':
        print("summing passage vectors...")
        p_vectors = (p_vectors_1 + p_vectors_2)
    else:
        raise ValueError(pooling)
    assert len(passages) == len(p_vectors)

    if save_embeddings:
        num_shards = 8
        len_per_shard = (len(p_vectors) // num_shards) + 1
        for i in range(num_shards):
            start_idx = i * len_per_shard
            end_idx = (i + 1) * len_per_shard
            with open(os.path.join(output_dir, f'reps_000{i}.pkl'), 'wb') as ouf:
                # weird bug where torch.float32 takes 32 bytes..
                pickle.dump(
                    torch.tensor(p_vectors[start_idx:end_idx].numpy()),
                    ouf,
                    protocol=4
                )

    del q_vectors_1_list
    del q_vectors_2_list
    del p_vectors_1
    del p_vectors_2
    print("building index...")
    index = build_index(p_vectors)

    gc.collect()
    for i, query_emb_name in enumerate(query_emb_names):
        questions = questions_list[i]
        retrieval_output_path = os.path.join(
            output_dir, output_filenames[i]
        )
        q_vectors = q_vectors_list[i]
        print("performing dense retrieval on", query_emb_name)
        # perform dense retrieval
        results = dense_search(questions, q_vectors, passages, index, topk)

        print("outputing results to", retrieval_output_path)
        os.makedirs(os.path.dirname(retrieval_output_path), exist_ok=True)
        with open(retrieval_output_path, 'w') as ouf:
            json.dump(results, ouf, indent=4)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_1_emb_dir", type=str, required=True)
    parser.add_argument("--model_2_emb_dir", type=str, required=True)
    parser.add_argument("--tsv_passages_path", type=str, required=True)
    parser.add_argument(
        "--jsonl_dataset_paths",
        nargs='+',
        help="paths to the JSONL dataset files; one for each dataset",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="directory for SPAR retrieval results, and optionally SPAR embeddings if save_embeddings is True"
    )
    parser.add_argument(
        "--save_embeddings",
        action='store_true',
        help="save the concatenated embeddings if True"
    )
    parser.add_argument(
        "--pred_filenames", nargs='+',
        default=[
            "nq_test.json",
            "squad1_test.json",
            "trivia_test.json",
            "webq_test.json",
            "trec_test.json",
        ],
        help="filenames to the JSON prediction files; one for each dataset"
    )
    parser.add_argument(
        "--query_reps_filenames", nargs='+',
        default=[
            "query_reps_nq_test.pkl",
            "query_reps_squad1_test.pkl",
            "query_reps_trivia_test.pkl",
            "query_reps_webq_test.pkl",
            "query_reps_trec_test.pkl",
        ],
        help="filenames to the query embedding files; one for each dataset"
    )
    parser.add_argument(
        "--weights", nargs="+", type=float,
        help="concat weights; one for each dataset"
    )
    parser.add_argument(
        "--topk", type=int, default=100,
        help="top-k retrieval results will be saved in the output."
    )
    parser.add_argument(
        "--pooling", type=str, default='concat',
        help="How to combine the vectors from the two models (default: concat)"
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    assert (
        len(args.pred_filenames)
        == len(args.query_reps_filenames)
        == len(args.jsonl_dataset_paths)
        == len(args.weights)
    )

    run_spar_retrieval(
        jsonl_dataset_paths=args.jsonl_dataset_paths,
        tsv_passages_path=args.tsv_passages_path,
        ctx_embeddings_dir_1=args.model_1_emb_dir,
        ctx_embeddings_dir_2=args.model_2_emb_dir,
        output_filenames=args.pred_filenames,
        query_emb_names=args.query_reps_filenames,
        output_dir=args.output_dir,
        save_embeddings=args.save_embeddings,
        weights=args.weights,
        topk=args.topk,
        pooling=args.pooling,
    )
