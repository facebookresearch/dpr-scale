# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import gzip
import json
import logging
import multiprocessing as mp
import os
from typing import List

from nltk.tokenize import word_tokenize
from tqdm import tqdm

from sentence_splitter import split_text_into_sentences


def get_logger():
    log_format = "[%(asctime)s] [%(levelname)s]: %(message)s"
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--doc_dir", type=str, default="/datasets01/CC-NEWS/022719/json/"
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    return parser


def get_lines(file_path_list: List[str], debug):
    for file_path in file_path_list:
        with gzip.open(file_path, "rt") as infile:
            for i, line in enumerate(infile):
                yield line
                if i == 10 and debug:
                    break


def process_json_line(line_str: str):
    line = json.loads(line_str)
    if (
        line["language"] != "en"
        or line["text"] is None
        or line["title"] is None
        or line["url"] is None
    ):
        return None
    sentences = split_text_into_sentences(line["text"], language="en")
    words = word_tokenize(line["text"])
    return line["url"], len(sentences), len(words)


def process_cc_news_files(files, workers, debug=False):
    url_dict = collections.defaultdict(int)
    num_samples = 0
    total_num_sents = 0
    total_num_words = 0

    with mp.Pool(processes=workers) as pool:
        for tup in tqdm(
            pool.imap_unordered(
                process_json_line, get_lines(files, debug), chunksize=1000
            )
        ):
            if tup is None:
                continue
            url, num_sents, num_words = tup

            url_dict[url] += 1
            total_num_sents += num_sents
            total_num_words += num_words
            num_samples += 1

    return num_samples, url_dict, total_num_sents, total_num_words


def main(args, logger):
    """
    No hard negative sampling done. Only positive samples given a passage are prepared.
    TODO: One way would be to get other passages from the same article as negatives.
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
    (
        num_samples,
        url_dict,
        total_num_sents,
        total_num_words,
    ) = process_cc_news_files(files, workers, args.debug)
    logger.info(f"{num_samples} samples were found")
    logger.info(f"{len(url_dict)} URLs were found")
    logger.info(
        f"{total_num_sents/num_samples} is the avg number of sentences"
    )
    logger.info(f"{total_num_words/num_samples} is the avg number of words")


if __name__ == "__main__":
    main(get_parser().parse_args(), get_logger())
