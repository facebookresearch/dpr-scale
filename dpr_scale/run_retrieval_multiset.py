# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import argparse
import logging
import os
from typing import Dict, List
# @manual=//faiss/python:pyfaiss
import faiss
import glob
import pickle
import pathlib
from dpr_scale.datamodule.dpr import CSVDataset, QueryCSVDataset
from dpr_scale.utils.utils import PathManager
try:
    # dummy import to make Manifold paths happy
    from pytext import fb  # noqa
except ImportError:
    print("Failed to import pytext. Ignore this if in OSS.")

import ujson as json


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
        "--query_emb_paths",
        type=str,
        nargs='+',
        default=['query_reps.pkl'],
        help="list of query emb files"
    )
    parser.add_argument(
        "--questions_tsv_paths",
        type=str,
        nargs='+',
        default="list of dataset files",
    )
    parser.add_argument(
        "--passages_tsv_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--output_json_paths",
        type=str,
        nargs='+',
        default="list of output filenames",
    )
    return parser


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
        with PathManager.open(fname, 'rb') as f:
            vector = pickle.load(f)  # noqa
            if not index:
                index = faiss.IndexFlatIP(vector.size()[1])
            print(f"Adding {vector.size()} vectors from {fname}")
            index.add(vector.numpy())
    return index


def main(args, logger):
    # Temp patch for datamodule refactoring
    logger.info(args.__dict__)

    assert (
        len(args.query_emb_paths)
        == len(args.questions_tsv_paths)
        == len(args.output_json_paths)
    )
    # index all passages
    print("Loading passage vectors.")
    local_ctx_embeddings_dir = PathManager.get_local_path(
        args.ctx_embeddings_dir
    )
    input_paths = sorted(glob.glob(
        os.path.join(local_ctx_embeddings_dir, "reps_*")
    ))
    index = build_index(input_paths)

    # load all passages:
    print(f"Loading passages from {args.passages_tsv_path}")
    ctxs = CSVDataset(args.passages_tsv_path)

    for questions_tsv_path, query_emb_path, output_json_path in zip(
        args.questions_tsv_paths, args.query_emb_paths, args.output_json_paths
    ):
        # load questions file
        print(f"Loading questions file {questions_tsv_path}")
        questions = QueryCSVDataset(questions_tsv_path)

        # reload question embeddings
        print("Loading question vectors.")
        if not query_emb_path:
            query_emb_path = os.path.join(
                args.ctx_embeddings_dir, "query_reps.pkl"
            )
        with PathManager.open(
            query_emb_path, "rb"
        ) as f:
            q_repr = pickle.load(f)  # noqa

        print("Retrieving results...")
        scores, indexes = index.search(q_repr.numpy(), 100)

        # write output file
        print("Merging results...")
        results = merge_results(ctxs, questions, indexes, scores)

        print(f"Writing output to {output_json_path}")
        pathlib.Path(output_json_path).parent.mkdir(
            parents=True, exist_ok=True
        )
        with PathManager.open(output_json_path, "w") as g:
            g.write(json.dumps(results, indent=4))
            g.write("\n")


if __name__ == "__main__":
    main(get_parser().parse_args(), get_logger())
