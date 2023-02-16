# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from tqdm import tqdm
import argparse
import json
from collections import defaultdict

def read_query(file):
    qid2query = {}
    with open(file, 'r') as fin:
        for line in tqdm(fin, desc='read queries'):
            qid, query = line.strip().split('\t')
            qid2query[int(qid)] = query 
    return qid2query


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio_of_dev", type=float, default=0.01)
    parser.add_argument("--query_file_path", type=str, required=True)
    parser.add_argument("--qidpidtriples_path", type=str, required=True)
    parser.add_argument("--json_output_dir", type=str, required=True)
    args = parser.parse_args()

    qid2query = read_query(args.query_file_path)
    train_out = open(os.path.join(args.json_output_dir, 'official_train.jsonl'), 'w')
    dev_out = open(os.path.join(args.json_output_dir, 'dev.jsonl'), 'w')
    qid2postive = defaultdict(set)
    qid2negative = defaultdict(set)

    with open(args.qidpidtriples_path, 'r') as fin:
        for line in tqdm(fin, desc='read qidpidtriples'):
            qid, positive_pid, negative_pid = line.strip().split('\t')
            qid = int(qid)
            positive_pid = int(positive_pid)
            negative_pid = int(negative_pid)
            qid2postive[qid].add(positive_pid)
            qid2negative[qid].add(negative_pid)
    
    for i, qid in tqdm(enumerate(qid2postive), total=len(qid2postive), desc='write train and dev data'):
        output_dict = {}
        output_dict['query_id'] = qid
        output_dict['question'] = qid2query[qid]

        positive_ctxs = []
        for positive_pid in qid2postive[qid]:
            positive_ctxs.append({'docidx': positive_pid}) # msmarco docid equal to the position in the corpus (start from 0)
        output_dict['positive_ctxs'] = positive_ctxs

        hard_negative_ctxs = []
        for negative_pid in qid2negative[qid]:
            hard_negative_ctxs.append({'docidx': negative_pid}) # msmarco docid equal to the position in the corpus (start from 0)
        output_dict['hard_negative_ctxs'] = hard_negative_ctxs

        if i < int(len(qid2postive) * args.ratio_of_dev):
            dev_out.write(json.dumps(output_dict) + '\n')
        else:
            train_out.write(json.dumps(output_dict) + '\n')

    dev_out.close()
    train_out.close()


if __name__ == "__main__":
	main()
    