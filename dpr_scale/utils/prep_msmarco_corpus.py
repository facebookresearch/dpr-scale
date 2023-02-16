import argparse
import logging
import os
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
    parser.add_argument("--debug", action="store_true")
    return parser


def process_msmarco(
    doc_path: str,
    output_dir_path: str,
    debug: bool = False,
) -> int:
    num_samples = 0
    outfile_path = os.path.join(output_dir_path, "msmarco_corpus.tsv")
    
    with jsonlines.open(doc_path) as reader, open(outfile_path, "w") as outfile:    
        outfile.write("id\ttext\ttitle\n")
        for item in reader: 
            outfile.write("{docid}\t{text}\t{title}\n".format(**item))
            num_samples += 1
            if debug and num_samples == 20:
                break
        return num_samples

def main(args, logger):
    # PYTHONPATH=. python dpr_scale/utils/prep_msmarco_corpus.py  --doc_path <corpus file> --output_dir_path <output dir path>

    logger.info(args.__dict__)
    num_samples = process_msmarco(
        args.doc_path,
        args.output_dir_path,
        args.debug,
    )
    logger.info(f"{num_samples} were written to {args.output_dir_path}")


if __name__ == "__main__":
    main(get_parser().parse_args(), get_logger())
