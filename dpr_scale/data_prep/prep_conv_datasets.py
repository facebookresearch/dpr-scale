# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import pandas as pd
import logging
import os
import ujson
from tqdm import tqdm


def get_logger():
    log_format = "[%(asctime)s] [%(levelname)s]: %(message)s"
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["dstc7", "ubuntu2"])
    # The files are extracted from the tar ball here
    # https://github.com/facebookresearch/ParlAI/blob/master/parlai/tasks/dstc7/build.py\
    # https://github.com/facebookresearch/ParlAI/blob/master/parlai/tasks/dstc7/build.py
    parser.add_argument("--in_file_path", type=str)
    parser.add_argument("--out_file_path", type=str)
    return parser


def get_question(messages_so_far):
    return " ".join(
        [
            m["speaker"].lstrip("participant_") + ": " + m["utterance"] + "\n"
            for m in messages_so_far
        ]
    )


def get_ctx(text):
    return {"text": text, "title": ""}


def get_pos_ctxs(options_for_correct_answers):
    pos_ctxs = []
    pos_ctx_ids = set()
    for m in options_for_correct_answers:
        pos_ctxs.append(get_ctx(m["utterance"]))
        pos_ctx_ids.add(m["candidate-id"])
    return pos_ctxs, pos_ctx_ids


def get_neg_ctxs(options_for_next, pos_ctx_ids):
    neg_ctxs = [
        get_ctx(m["utterance"])
        for m in options_for_next
        if m["candidate-id"] not in pos_ctx_ids
    ]
    return neg_ctxs


def prep_dpr_dstc7(infile, outfile):
    skipped = 0
    with open(infile) as fin, open(outfile, "w") as fout:
        json_obj = ujson.load(fin)
        for line in tqdm(json_obj):
            if "options-for-correct-answers" in line:
                question = get_question(line["messages-so-far"])
                pos_ctxs, pos_ctx_ids = get_pos_ctxs(
                    line["options-for-correct-answers"]
                )
                neg_ctxs = get_neg_ctxs(line["options-for-next"], pos_ctx_ids)
                out_json = ujson.dumps(
                    {
                        "question": question,
                        "answers": [],
                        "positive_ctxs": pos_ctxs,
                        "hard_negative_ctxs": neg_ctxs,
                    }
                )
                fout.write(f"{out_json}\n")
            else:
                skipped += 1
    print(f"{infile}: {skipped}")


def prep_dpr_ubuntuv2(infile, outfile):
    num_samples = 0
    df = pd.read_csv(infile)
    is_train = os.path.basename(infile).rstrip(".csv") == "train"
    with open(outfile, "w") as fout:
        # Context,Ground Truth Utterance,Distractor_0,Distractor_1,...,Distractor_8
        for i, row in tqdm(df.iterrows(), total=len(df)):
            if is_train:  # train only has +ve samples.
                question = row["Context"]
                pos_ctxs = [get_ctx(row["Utterance"])]
                neg_ctxs = []
            else:
                question = row["Context"]
                pos_ctxs = [get_ctx(row["Ground Truth Utterance"])]
                neg_ctxs = [get_ctx(row[f"Distractor_{i}"]) for i in range(9)]
                assert len(neg_ctxs) == 9, (len(neg_ctxs), row)
            out_json = ujson.dumps(
                {
                    "question": question,
                    "answers": [],
                    "positive_ctxs": pos_ctxs,
                    "hard_negative_ctxs": neg_ctxs,
                }
            )
            fout.write(f"{out_json}\n")
            num_samples += 1
    return num_samples


def main(args, logger):
    logger.info(args.__dict__)
    if not args.in_file_path or not args.out_file_path:
        logger.error("You must provide paths to input and output files.")
        return
    if not os.path.exists(args.in_file_path):
        logger.error(f"{args.in_file_path} doens't exist")
        return
    os.makedirs(os.path.dirname(args.out_file_path), exist_ok=True)

    prep_func = (
        prep_dpr_dstc7 if args.dataset == "dstc7" else prep_dpr_ubuntuv2
    )
    num_samples = prep_func(args.in_file_path, args.out_file_path)
    logger.info(f"{num_samples} were written to {args.out_file_path}")


if __name__ == "__main__":
    main(get_parser().parse_args(), get_logger())
