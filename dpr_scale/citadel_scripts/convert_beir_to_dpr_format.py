# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import json
import os
import sys
import csv
from tqdm import tqdm
from beir import util
import shutil

def load_transfrom_corpus(input_path, output_path):
    corpus = ["id\ttext\ttitle\n"]
    docid_strs = []
    num_lines = sum(1 for i in open(input_path, 'rb'))
    docid = 0
    with open(input_path, encoding='utf8') as fIn:
        for line in tqdm(fIn, total=num_lines):
            line = json.loads(line)
            docid_str = line.get("_id").replace('\n', ' ').replace('\t', ' ')
            text = line.get("text").replace('\n', ' ').replace('\t', ' ')
            title = line.get("title").replace('\n', ' ').replace('\t', ' ')
            docid_strs.append(f"{docid_str}\n")
            corpus.append(f"{docid}\t{text}\t{title}\n")
            docid += 1
    with open(output_path, "w") as f:
        f.writelines(corpus)
    
    output_dir = os.path.dirname(output_path)
    output_path = os.path.join(output_dir, "index2docid.tsv")
    with open(output_path, "w") as f:
        f.writelines(docid_strs)
        
def load_transfrom_queries(input_path, output_path, qrels):
    queries = []
    with open(input_path, encoding='utf8') as fIn:
        for line in fIn:
            line = json.loads(line)
            id, text = line.get("_id").strip(), line.get("text").strip()
            if id in qrels:
                queries.append(f"{id}\t{text}\n")
    with open(output_path, "w") as f:
        f.writelines(queries)

def _load_qrels(qrels_file):
    qrels = {}
    reader = csv.reader(open(qrels_file, encoding="utf-8"), 
                        delimiter="\t", quoting=csv.QUOTE_MINIMAL)
    next(reader)
    for row in reader:
        query_id, corpus_id, score = row[0], row[1], int(row[2])
        if query_id not in qrels:
            qrels[query_id] = {corpus_id: score}
        else:
            qrels[query_id][corpus_id] = score
    return qrels

def main():
    dataset = sys.argv[1]
    output_path = sys.argv[2]

    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    out_dir = os.path.join(output_path, "datasets")
    data_path = util.download_and_unzip(url, out_dir)
    os.makedirs(os.path.join(data_path, "dpr-scale"), exist_ok=True)
    load_transfrom_corpus(os.path.join(data_path, "corpus.jsonl"), os.path.join(data_path, "dpr-scale", "corpus.tsv"))
    qrels = _load_qrels(os.path.join(data_path, "qrels/test.tsv"))
    load_transfrom_queries(os.path.join(data_path, "queries.jsonl"), os.path.join(data_path, "dpr-scale", "queries.tsv"), qrels)
    shutil.copy(os.path.join(data_path, "qrels/test.tsv"), os.path.join(data_path, "dpr-scale"))

if __name__ == "__main__":
    main()
