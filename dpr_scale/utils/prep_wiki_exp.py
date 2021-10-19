# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import logging
import multiprocessing as mp
import os
import random
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

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
        # default="/private/home/vladk/data/wikipedia/wiki_passages/psgs_w100.tsv",
    )
    parser.add_argument(
        "--doc_dict_path",
        type=str,
        # default="/checkpoint/kushall/data/wikipedia/psgs_w100_doc_dict.npy",
    )
    parser.add_argument(
        "--output_dir_path",
        type=str,
        default="/checkpoint/kushall/data/wikipedia",
    )
    parser.add_argument(
        "--dev_pct",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--base",
        type=float,
        default=1.2,
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
    base: float
) -> Iterable[Tuple[List[str], str, int, int]]:
    for title, passage_list in doc_dict.items():
        # pmf = [1/(base*(i+1)) for i in range(passage_list)]
        for i, (passage, id) in enumerate(passage_list):
            passage_sents = split_text_into_sentences(passage, language="en")
            num_queries = max(
                1, round(len(passage_sents) / (base * (i + 1)))
            )
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
    dev_pct: float,
    base: float,
    logger,
    debug: bool = False,
) -> int:
    num_samples = 0

    train_file_path = os.path.join(output_dir_path, f"wiki_ict_exp_train_base{base}.jsonl")
    dev_file_path = os.path.join(output_dir_path, f"wiki_ict_exp_dev_base{base}.jsonl")
    doc_dict = (
        prep_wiki_psgs_w100(doc_path, debug)
        if not doc_dict_path
        else np.load(doc_dict_path, allow_pickle=True).item()
    )
    logger.info(f"Loaded {len(doc_dict)} in doc_dict")
    
    with mp.Pool(processes=workers) as pool, open(
        train_file_path, "w"
    ) as train_file, open(dev_file_path, "w") as dev_file:
        for task_out in tqdm(
            pool.imap_unordered(
                process_passage,
                get_lines(doc_dict, base),
                chunksize=10000,
            )
        ):
            # for task_out in tqdm(
            #     map(
            #         process_passage,
            #         get_lines(doc_dict),
            #     )
            # ):
            if task_out is None:
                continue

            for (
                question_pos,
                question,
                passage,
                title,
                passage_id,
            ) in task_out:
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
    # PYTHONPATH=. python dpr_scale/utils/prep_wiki_exp.py --workers 16 --doc_path /private/home/vladk/data/wikipedia/wiki_passages/psgs_w100.tsv
    # PYTHONPATH=. python dpr_scale/utils/prep_wiki_exp.py --workers 16 --doc_dict_path /checkpoint/kushall/data/wikipedia/psgs_w100_doc_dict.npy

    logger.info(args.__dict__)
    logger.info(
        f"Train split = {100-args.dev_pct}% and dev split = {args.dev_pct}%"
    )

    num_samples = process_wiki_ict(
        args.doc_path,
        args.doc_dict_path,
        args.workers,
        args.output_dir_path,
        args.dev_pct,
        args.base,
        logger,
        args.debug,
    )
    logger.info(f"{num_samples} were written to {args.output_dir_path}")


if __name__ == "__main__":
    main(get_parser().parse_args(), get_logger())
