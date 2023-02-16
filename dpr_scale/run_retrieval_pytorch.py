# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.
# This is revised from https://github.com/facebookresearch/dpr-scale/blob/main/dpr_scale/run_retrieval_fb.py for research evaluation purpose.
# Without installing Faiss, we directly use pytorch for brute-force inner-product search. Some additional functions are supported:
# (1) If index is too large for single GPU search, you can divide the index into several segments. 
#     For example, if you have 4 segments of index, you can set --shard to be 4; then, it will search the 4 segments sequentially and merge
#     the results in a runfile. You can also set --shard to 2; then, it will search the first two indices and then the other two indices;
#     then, merge the results. But you cannot set --shard to 3 in this scenario. 
# (2) Support --trec_format query input (qid\tquery_text\n) and output format (qid Q0 docid rank scores runname) for trec evaluation.
# (3) Support --ignore_identical_ids used in BEIR Arguana and Quora datasets.

import argparse
import logging
import os
from typing import Dict, List
# @manual=//faiss/python:pyfaiss
import glob
import pickle
import pathlib
from dpr_scale.datamodule.dpr import CSVDataset, QueryCSVDataset, QueryTSVDataset
from dpr_scale.utils.utils import PathManager
try:
    # dummy import to make Manifold paths happy
    from pytext import fb  # noqa
except ImportError:
    print("Failed to import pytext. Ignore this if in OSS.")

import ujson as json
import numpy as np
from tqdm import tqdm
import torch
import time

def get_logger():
    log_format = "[%(asctime)s] [%(levelname)s]: %(message)s"
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ctx_embeddings_dir",
        type=str,
        default="",
    )
    parser.add_argument(
        "--query_emb_path",
        type=str,
        default="",
        help="if left empty, will use <ctx_embeddings_dir>/query_reps.pkl"
    )
    parser.add_argument(
        "--questions_tsv_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--passages_tsv_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--output_runfile_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--shard",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--trec_format",
        action='store_true'
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="dpr",
    )
    parser.add_argument(
        "--ignore_identical_ids",
        action='store_true',
        help="this is used for BEIR Arguana and Quora datasets"
    )
    return parser


def merge_results(
    passages: Dict,
    questions: List,
    top_doc_ids: List,
    scores_list: List,
    trec_format: bool,
):
    # join passages text with the result ids, their questions
    merged_data = []
    assert len(top_doc_ids) == len(questions) == len(scores_list)
    for i, question, doc_ids, scores in zip(range(len(questions)), questions, top_doc_ids, scores_list):
        if not trec_format:
            ctxs = [
                {
                    "id": passages[id]["id"],
                    "title": passages[id]["title"],
                    "text": passages[id]["text"],
                    "score": float(score),
                }
                for id, score in zip(doc_ids, scores)
            ]
        else: # output trec format
            ctxs = []
            for id, score in zip(doc_ids, scores):
                try:
                    ctxs.append({"id": passages[id]["id"], "score": float(score)})
                except:
                    continue # avoid error in beir evaluation (lines are empty)
                    
        merged_data.append(
            {
                "question": question["question"],
                "answers": question["answers"] if "answers" in question else [],
                "ctxs": ctxs,
                "id": question.get("id", i),
            }
        )

    return merged_data


def search_index(query_embs, corpus_embs, batch, topk):

    all_scores = np.zeros((query_embs.shape[0], topk))
    all_results =  np.zeros((query_embs.shape[0], topk))

    start_time = time.time()

    if batch > query_embs.shape[0]:
        scores = torch.einsum('ik,jk->ij',(query_embs.to(torch.float16).cuda(0), corpus_embs))
        sort_scores, sort_candidates = torch.topk(scores, dim=-1, k=topk)

        all_scores = sort_scores.cpu()
        all_results = sort_candidates.cpu()
    else:
        for i in tqdm(range(query_embs.shape[0]//batch) ,total=query_embs.shape[0]//batch, desc='Search with batch size {}'.format(batch)):
            scores = torch.einsum('ik,jk->ij',(query_embs[i*batch:(i+1)*batch].to(torch.float16).cuda(0), corpus_embs))
            sort_scores, sort_candidates = torch.topk(scores, dim=-1, k=topk)

            all_results[i*batch:(i+1)*batch, :] = sort_candidates.cpu()
            all_scores[i*batch:(i+1)*batch, :] = sort_scores.cpu()
            
            del scores
            del sort_scores

        scores = torch.einsum('ik,jk->ij',(query_embs[(i+1)*batch:].to(torch.float16).cuda(0), corpus_embs))
        sort_scores, sort_candidates = torch.topk(scores, dim=-1, k=topk)

        all_scores[(i+1)*batch:, :] = sort_scores.cpu()
        all_results[(i+1)*batch:, :] = sort_candidates.cpu()

    del scores
    del sort_scores
    time_per_query = (time.time() - start_time)/query_embs.shape[0]
    print('Retrieving {} queries ({:0.3f} s/query)'.format(query_embs.shape[0], time_per_query))

    return all_scores, all_results

def build_index(paths):
    index = None
    vector_num = 0
    for fname in paths:
        with PathManager.open(fname, 'rb') as f:
            vector = torch.tensor(pickle.load(f))
            if index is None:
                index = torch.zeros((vector.shape[0]*len(paths)), vector.shape[1])

            index[vector_num: (vector_num + vector.shape[0]), :] = vector
            vector_num += vector.shape[0]
            print(f"Adding {vector.size()} vectors from {fname}")
    return index.to(torch.float16).cuda(0)

def main(args, logger):

    # Temp patch for datamodule refactoring
    logger.info(args.__dict__)
    
    # index all passages and search with shards
    local_ctx_embeddings_dir = PathManager.get_local_path(
        args.ctx_embeddings_dir
    )
    input_paths = sorted(glob.glob(
        os.path.join(local_ctx_embeddings_dir, "reps_*")
    ))

    assert len(input_paths) % args.shard == 0, "Invalid Shard number"
    file_num = len(input_paths) // args.shard
    offset = 0
    all_scores, all_indexes = None, None

    for shard_num in range(args.shard):
        index = build_index(input_paths[shard_num*file_num:(shard_num+1)*file_num])

        if shard_num==0: #avoid loading again
            print("Loading question vectors.")
            with open(
                args.query_emb_path, "rb"
            ) as f:
                q_repr = torch.tensor(pickle.load(f))
        
        print("Retrieving results...")
        scores_partial, indexes_partial = search_index(q_repr, index, args.batch, args.topk)
        total_doc_num = len(index)
        del index

        if all_scores is None:
            all_scores = np.zeros((scores_partial.shape[0], args.topk * args.shard))
            all_indexes = np.zeros((indexes_partial.shape[0], args.topk * args.shard))
        all_scores[:, (shard_num * args.topk):((shard_num + 1) * args.topk)] = scores_partial
        all_indexes[:, (shard_num * args.topk):((shard_num + 1) * args.topk)] = indexes_partial + offset
        offset += total_doc_num

    # load questions file
    print(f"Loading questions file {args.questions_tsv_path}")
    if args.trec_format:
        questions = QueryTSVDataset(args.questions_tsv_path)
    else:
        questions = QueryCSVDataset(args.questions_tsv_path)

    quesion_list = []
    for question in questions:
        quesion_list.append(question)
    # load all passages:
    print(f"Loading passages from {args.passages_tsv_path}")
    ctxs = CSVDataset(args.passages_tsv_path)

    print(f"Writing output to {args.output_runfile_path}")
    pathlib.Path(args.output_runfile_path).parent.mkdir(
        parents=True, exist_ok=True
    )

    query_num = all_scores.shape[0]
    write_batch = 1000
    if (query_num) % (write_batch) == 0:
        batch_num = query_num // write_batch
    else:
        batch_num = query_num // write_batch + 1

    # write output file
    qa_results = []
    with PathManager.open(args.output_runfile_path, "w") as g:
        for i in tqdm(range(batch_num), total=batch_num, desc='Write data with batch size: {}'.format(write_batch)):

            if i == (batch_num - 1):
                scores = all_scores[i*write_batch:]
                indexes = all_indexes[i*write_batch:]
                batch_questions = quesion_list[i*write_batch:]
            else:
                scores = all_scores[i*write_batch:(i+1)*write_batch]
                indexes = all_indexes[i*write_batch:(i+1)*write_batch]
                batch_questions = quesion_list[i*write_batch:(i+1)*write_batch]

            # we need to sort the score again if shard > 1
            if args.shard > 1:
                scores, idx = torch.topk(torch.tensor(scores).cuda(0), dim=-1, k=args.topk)
                indexes = torch.gather(torch.tensor(indexes).cuda(0), 1, idx)
                scores = scores.detach().cpu().numpy()
                indexes = indexes.detach().cpu().numpy()

            results = merge_results(ctxs, batch_questions, indexes, scores, args.trec_format)

            if not args.trec_format:
                qa_results += results
            else:
                for result in results:
                    qid = result['id']
                    for i, ctx in enumerate(result['ctxs']):
                        rank = i + 1
                        if args.ignore_identical_ids and qid == ctx['id']:
                            continue
                        else:
                            g.write('{} Q0 {} {} {} {}\n'.format(qid, ctx['id'], rank, ctx['score'], args.run_name))
                        
        if not args.trec_format:
            g.write(json.dumps(qa_results, indent=4))
            g.write("\n")

if __name__ == "__main__":
    main(get_parser().parse_args(), get_logger())
