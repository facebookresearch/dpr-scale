import argparse
import logging
import os
import random
import ujson
import jsonlines


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
        "--output_dir_path",
        type=str,
        default="/fsx/mhli/msmarco_passage/",
    )
    parser.add_argument(
        "--dev_pct",
        type=float,
        default=0.01,
    )
    parser.add_argument("--debug", action="store_true")
    return parser


def process_msmarco(
    doc_path: str,
    output_dir_path: str,
    dev_pct: float = None,
    debug: bool = False,
) -> int:
    num_samples = 0
    train_file_path = os.path.join(output_dir_path, "msmarco_exp_train.jsonl")
    dev_file_path = os.path.join(output_dir_path, "msmarco_exp_dev.jsonl")
    
    with jsonlines.open(doc_path) as reader, open(train_file_path, "w") as train_file, open(dev_file_path, "w") as dev_file:    
        for item in reader: 
            out_json = ujson.dumps(
                {   "dataset": "msmarco_passages",
                    "question_id": item["query_id"],
                    "question": item["query"],
                    "answers": [],
                    "positive_ctxs": [
                        {
                            "text": psg["text"],
                            "title": psg["title"],
                            "score": 1000,
                            "title_score": 1,
                            "passage_id": psg["docid"],
                        }
                    for psg in item["positive_passages"]],
                    "hard_negative_ctxs": [
                        {
                            "text": psg["text"],
                            "title": psg["title"],
                            "score": 1000,
                            "title_score": 1,
                            "passage_id": psg["docid"],
                        }
                    for psg in item["negative_passages"]],
                }
            )
            outfile = train_file if random.random() > dev_pct else dev_file
            outfile.write(f"{out_json}\n")

            num_samples += 1
            if debug and num_samples == 20:
                break
        return num_samples

def main(args, logger):
    # PYTHONPATH=. python dpr_scale/utils/prep_msmarco_exp.py --doc_path <train file> --output_dir_path <output dir path>

    logger.info(args.__dict__)
    logger.info(
        f"Train split = {100-args.dev_pct}% and dev split = {args.dev_pct}%"
    )
    num_samples = process_msmarco(
        args.doc_path,
        args.output_dir_path,
        args.dev_pct,
        args.debug,
    )
    logger.info(f"{num_samples} were written to {args.output_dir_path}")


if __name__ == "__main__":
    main(get_parser().parse_args(), get_logger())
