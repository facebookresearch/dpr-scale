# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import collections
import logging
import multiprocessing as mp
import random
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

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
    )
    parser.add_argument(
        "--doc_dict_path",
        type=str,
    )
    parser.add_argument(
        "--output_dir_path",
        type=str,
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    return parser


def get_random_query(passage_sents: List[str]) -> Tuple[str, int, List[str]]:
    query_pos = random.randint(0, len(passage_sents) - 1)
    query = passage_sents[query_pos]
    return query, query_pos


def get_ict_data(
    passage_sents: List[str], num_queries=1
) -> Tuple[int, str, str]:
    query_pos_set = set()
    num_queries_produced = 0
    while num_queries_produced < num_queries:
        query, query_pos = get_random_query(passage_sents)
        if query_pos in query_pos_set:
            continue
        query_pos_set.add(query_pos)
        num_queries_produced += 1
        yield (
            query_pos,
            query,
            " ".join(
                [
                    sent
                    for i, sent in enumerate(passage_sents)
                    if i != query_pos
                ]
            ),
        )


def process_passage(
    tup: Tuple[List[str], str, int, int]
) -> Tuple[str, str, str, str, str]:
    passage_sents, title, id, num_sents = tup
    ict_data_list = []
    for query_pos, query, passage in get_ict_data(passage_sents, num_sents):
        ict_data_list.append((query_pos, query, passage, title, id))
    return ict_data_list


def get_lines(
    doc_dict: Dict[str, List[str]],
) -> Iterable[Tuple[List[str], str, int, int]]:
    for title, passage_list in doc_dict.items():
        for passage, id in passage_list:
            passage_sents = split_text_into_sentences(passage, language="en")
            num_queries = len(passage_sents)
            yield (passage_sents, title, id, num_queries)


def prep_wiki_psgs_w100(file_path: str, debug: bool):
    doc_dict = collections.defaultdict(list)
    df = pd.read_csv(file_path, sep="\t")
    # Header: id,text,title
    for i, row in df.iterrows():
        passage = row["text"]
        title = row["title"]
        doc_dict[title].append((passage, row["id"]))
        if debug and i == 500:
            break
    return doc_dict


def process_wiki_ict(
    doc_path: str,
    doc_dict_path: str,
    workers: int,
    output_dir_path: str,
    logger,
    debug: bool = False,
) -> int:
    num_samples = 0

    train_file_path = output_dir_path
    doc_dict = (
        prep_wiki_psgs_w100(doc_path, debug)
        if not doc_dict_path
        else np.load(doc_dict_path, allow_pickle=True).item()
    )
    logger.info(f"Loaded {len(doc_dict)} in doc_dict")
    
    with mp.Pool(processes=workers) as pool, open(
        train_file_path, "w"
    ) as train_file:
        for task_out in tqdm(
            pool.imap_unordered(
                process_passage,
                get_lines(doc_dict),
                chunksize=10000,
            )
        ):
            if task_out is None:
                continue

            for (
                question_pos,
                question,
                passage,
                title,
                passage_id,
            ) in task_out:
                train_file.write("{}\t{}\n".format(num_samples, question))
                num_samples += 1

                if debug and num_samples == 20:
                    break

    return num_samples


def main(args, logger):

    logger.info(args.__dict__)
    logger.info(
        "Cropping sentences from {} as {}".format(args.doc_path, args.output_dir_path)
    )

    num_samples = process_wiki_ict(
        args.doc_path,
        args.doc_dict_path,
        args.workers,
        args.output_dir_path,
        logger,
        args.debug,
    )
    logger.info(f"{num_samples} were written to {args.output_dir_path}")


if __name__ == "__main__":
    main(get_parser().parse_args(), get_logger())
