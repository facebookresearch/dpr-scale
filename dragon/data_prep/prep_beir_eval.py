# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from tqdm import tqdm
import argparse
import json
import os
from typing import List, Optional

def json_to_tsv(
    in_path: str, 
    out_path: str, 
    meta_list: List[str], 
    head: Optional[List[str]] = None
):
    
    with open(out_path, 'w') as fout:
        with open(in_path, 'r') as fin:
            for i, line in tqdm(enumerate(fin)):
                content = json.loads(line)
                if (i == 0) and (head is not None):
                    # write head
                    fout.write('\t'.join(head) + '\n')

                text_list = []
                for item in meta_list:
                    if item == "text" or item == "title":
                        content[item] = ' '.join(content[item].split()) # avoid '\t' and '\n' in text and title to impact file reader
                    text_list.append(content[item])
                fout.write('\t'.join(text_list) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()
    
    qrel_tsv_path = os.path.join(args.data_dir, 'qrels', 'test.tsv')
    query_json_path = os.path.join(args.data_dir, 'queries.jsonl')
    query_tsv_path = os.path.join(args.data_dir, 'queries.test.tsv')
    corpus_json_path = os.path.join(args.data_dir, 'corpus.jsonl')
    corpus_tsv_path = os.path.join(args.data_dir, 'collection.tsv')
    
    print('output collection tsv')
    json_to_tsv(corpus_json_path, corpus_tsv_path, ["_id", "text", "title"], ["id", "text", "title"])

    print('output query tsv')
    json_to_tsv(query_json_path, query_tsv_path, ["_id", "text"])

    print('output qrel tsv')
    with open(os.path.join(args.data_dir, 'qrels.test.tsv'), 'w') as fout:
        with open(qrel_tsv_path, 'r') as fin:
            for i, line in tqdm(enumerate(fin)):
                if (i == 0): 
                    continue #skip head
                else:
                    qid, pid, rel = line.split('\t')
                    fout.write('{} {} {} {}'.format(qid, 0, pid, rel))
    
    
if __name__ == "__main__":
	main()
