# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import logging
import multiprocessing as mp
import os
import random
from typing import Iterable, List, Tuple

import ujson
from sentence_splitter import split_text_into_sentences
from tqdm import tqdm


def get_logger():
    log_format = "[%(asctime)s] [%(levelname)s]: %(message)s"
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--doc_path",
        type=str,
        default="/private/home/vladk/data/wikipedia/wiki_passages/psgs_w100.tsv",
    )
    parser.add_argument(
        "--output_dir_path",
        type=str,
        default="/checkpoint/kushall/data/wikipedia",
    )
    parser.add_argument(
        "--train_dev_split",
        type=str,
        default="[99.99,0.01]",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    return parser


def get_random_query(document: str) -> Tuple[str, int, List[str]]:
    sentences = split_text_into_sentences(document, language="en")
    query_pos = random.randint(0, len(sentences) - 1)
    query = sentences[query_pos]
    return query, query_pos, sentences


def get_ict_data(document: str) -> Tuple[int, str, str]:
    query, query_pos, sentences = get_random_query(document)
    return (
        query_pos,
        query,
        " ".join([sent for i, sent in enumerate(sentences) if i != query_pos]),
    )


def process_tsv_line(line_str: str) -> Tuple[str, str, str, str, str]:
    parts = line_str.rstrip().split("\t")
    query_pos, query, passage = get_ict_data(parts[1])

    title = " ".join(parts[2].split())
    return query_pos, query, passage, title, parts[0]


def get_lines(file_path: str) -> Iterable[str]:
    with open(file_path, "rt") as infile:
        for line in infile:
            yield line


def process_wiki_ict(
    file_path: str,
    workers: int,
    output_dir_path: str,
    train_pct: float,
    dev_pct: float,
    debug: bool = False,
) -> int:
    num_samples = 0

    train_file_path = os.path.join(output_dir_path, "wiki_ict_train.jsonl")
    dev_file_path = os.path.join(output_dir_path, "wiki_ict_dev.jsonl")

    with mp.Pool(processes=workers) as pool, open(
        train_file_path, "w"
    ) as train_file, open(dev_file_path, "w") as dev_file:
        for task_out in tqdm(
            pool.imap_unordered(
                process_tsv_line,
                get_lines(file_path),
                chunksize=10000,
            )
        ):
            if task_out is None:
                continue

            question_pos, question, passage, title, passage_id = task_out
            out_json = ujson.dumps(
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
                            "passage_id": passage_id,
                        }
                    ],
                    "hard_negative_ctxs": [],
                }
            )
            outfile = train_file if random.random() > dev_pct else dev_file
            outfile.write(f"{out_json}\n")

            num_samples += 1
            if debug and num_samples == 20:
                break

    return num_samples


def main(args, logger):
    """
    No hard negative sampling done.
    Only positive samples given a passage are prepared.
    """

    logger.info(args.__dict__)

    train_pct, dev_pct = eval(args.train_dev_split)
    logger.info(f"Train split = {train_pct}% and dev split = {dev_pct}%")

    num_samples = process_wiki_ict(
        args.doc_path,
        args.workers,
        args.output_dir_path,
        train_pct,
        dev_pct,
        args.debug,
    )
    logger.info(f"{num_samples} were written to {args.output_dir_path}")


if __name__ == "__main__":
    main(get_parser().parse_args(), get_logger())
