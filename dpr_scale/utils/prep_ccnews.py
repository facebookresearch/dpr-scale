# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import functools
import gzip
import logging
import multiprocessing as mp
import os
import random
from typing import Any, Iterable, List, Tuple

from tqdm import tqdm

import ujson
from sentence_splitter import split_text_into_sentences


def get_logger():
    log_format = "[%(asctime)s] [%(levelname)s]: %(message)s"
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", type=str, choices={"ict", "ict_chunked", "bfs", "all"}
    )
    parser.add_argument(
        "--doc_dir", type=str, default="/datasets01/CC-NEWS/022719/json/"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/checkpoint/kushall/data/cc_news/cc_news_ict.jsonl",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    return parser


def get_random_query(document: str) -> Tuple[str, int, List[str]]:
    sentences = split_text_into_sentences(document, language="en")
    query_pos = random.randint(0, len(sentences) - 1)
    query = sentences[query_pos]
    return query, query_pos, sentences


def split_document(document: str, passage_len: int) -> List[List[str]]:
    tokens = document.split()
    return [
        tokens[i: i + passage_len] for i in range(0, len(tokens), passage_len)
    ]


def get_ict_data(document: str, *args) -> List[Tuple[int, str, str]]:
    query, query_pos, sentences = get_random_query(document)
    return [
        (
            query_pos,
            query,
            " ".join(
                [sent for i, sent in enumerate(sentences) if i != query_pos]
            ),
        )
    ]


def get_ict_chunk_data(
    document: str, passage_len: int
) -> List[Tuple[int, Tuple[int, str, str]]]:
    chunks = split_document(document, passage_len)

    out_tuples = []
    for i, chunk in enumerate(chunks):
        ict_tup = get_ict_data(" ".join(chunk))[0]
        out_tuples.append((i, ict_tup))

    return out_tuples


def get_bfs_data(
    document: str, passage_len: int
) -> List[Tuple[str, int, int, str]]:
    chunks = split_document(document, passage_len)
    if len(chunks) <= 1:
        return ()

    query, query_pos, sentences = get_random_query(" ".join(chunks[0]))
    chunk_pos = random.randint(1, len(chunks) - 1)

    return [(query, query_pos, chunk_pos, " ".join(chunks[chunk_pos]))]


def get_task_func(task: str):
    if task == "ict":
        return get_ict_data
    elif task == "ict_chunked":
        return get_ict_chunk_data
    elif task == "bfs":
        return get_bfs_data
    else:
        raise Exception("Task = {task} is not supported yet.")


def get_task_out_json(
    task: str, task_tuple: Tuple[Any, ...], title: str, file_name: str
) -> str:
    if task == "ict":
        question_pos, question, passage = task_tuple
        passage_idx = 0  # Entire document is the passage
    elif task == "ict_chunked":
        passage_idx, ict_tup = task_tuple
        question_pos, question, passage = ict_tup
    elif task == "bfs":
        question, question_pos, passage_idx, passage = task_tuple

    return ujson.dumps(
        {
            "question": question,
            "question_pos": question_pos,
            "answers": [],
            "positive_ctxs": [
                {
                    "text": passage,
                    "title": title,
                    "score": 1000,
                    "title_score": 1,
                    "passage_id": f"{passage_idx}_{file_name}",
                }
            ],
            "hard_negative_ctxs": [],
        }
    )


def process_json_line(
    line_str: str, task: str
) -> Tuple[Tuple[Any, ...], str, str]:
    line = ujson.loads(line_str)
    if (
        line["language"] != "en"
        or line["text"] is None
        or line["title"] is None
    ):
        return None

    task_func = get_task_func(task)
    task_data = task_func(line["text"], 100)

    title = " ".join(line["title"].split())
    return task_data, title, line["filename"]


def get_lines(file_path_list: List[str]) -> Iterable[str]:
    for file_path in file_path_list:
        with gzip.open(file_path, "rt") as infile:
            for line in infile:
                yield line


def process_cc_news_files(
    task: str, file_path_list: List[str], workers: int, output_path: str
) -> int:
    num_samples = 0

    process_json_line_partial = functools.partial(process_json_line, task=task)
    with mp.Pool(processes=workers) as pool, open(output_path, "w") as outfile:
        for task_out in tqdm(
            pool.imap_unordered(
                process_json_line_partial,
                get_lines(file_path_list),
                chunksize=10000,
            )
        ):
            if task_out is None:
                continue

            out_task_tuples, title, file_name = task_out
            if not out_task_tuples:
                continue

            for task_tuple in out_task_tuples:
                out_json = get_task_out_json(
                    task, task_tuple, title, file_name
                )
                outfile.write(f"{out_json}\n")
                num_samples += 1

    return num_samples


def main(args, logger):
    """
    No hard negative sampling done.
    Only positive samples given a passage are prepared.
    """

    logger.info(args.__dict__)
    files = [
        os.path.join(dir_path, file_name)
        for (dir_path, dir_names, file_names) in os.walk(args.doc_dir)
        for file_name in file_names
    ]
    if args.debug:
        files = files[:2]

    workers = min(args.workers, len(files))
    logger.info(f"Number of workers = {workers}")
    num_samples = process_cc_news_files(
        args.task, files, workers, args.output_path
    )
    logger.info(f"{num_samples} were written to {args.output_path}")


if __name__ == "__main__":
    main(get_parser().parse_args(), get_logger())
