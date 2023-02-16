# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from tqdm import tqdm
import argparse
import json
from collections import defaultdict

def read_query(file):
    qid2query = {}
    with open(file, 'r') as fin:
        for i, line in tqdm(enumerate(fin), desc='read query file {}'.format(file)):
            qid, query = line.strip().split('\t')
            qid2query[qid] = query

    return qid2query


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_tsv_path", type=str, required=True)
    parser.add_argument("--trec_in_path", type=str, required=True)
    parser.add_argument("--json_out_path", type=str, required=True)
    args = parser.parse_args()

    qid2query = read_query(args.query_tsv_path)
    fout = open(args.json_out_path, 'w')
    qid2pid = defaultdict(list)
    qid2pidscore = defaultdict(list)

    with open(args.trec_in_path, 'r') as fin:
        for line in tqdm(fin):
            qid, _, pid, rank, score, _= line.strip().split(' ')
            if int(rank) > 50: # we only need top-50 samples to create training data
                continue
            qid2pid[qid].append(pid)
            qid2pidscore[qid].append(float(score))

    output_dict = {}
    for qid in tqdm(qid2pid, total=len(qid2pid), desc='write train data'):
        
        output_dict['query_id'] = qid
        output_dict['question'] = qid2query[qid]

        pid_list = qid2pid[qid]
        score_list = qid2pidscore[qid]
        positive_ctxs = []
        hard_negative_ctxs = []

        for pid, score in zip(pid_list[:10], score_list[:10]):
            # msmarco passage id is equal to its position in the corpus or we need a qid to docidx mapping
            positive_ctxs.append({'docidx': pid, 'relevance': score}) 
        
        for pid, score in zip(pid_list[45:50], score_list[45:50]):
            # msmarco passage id is equal to its position in the corpus or we need a qid to docidx mapping
            hard_negative_ctxs.append({'docidx': pid, 'relevance': score})
        
        output_dict['positive_ctxs'] = positive_ctxs
        output_dict['hard_negative_ctxs'] = hard_negative_ctxs
        if len(output_dict['positive_ctxs'])!=0:
            fout.write(json.dumps(output_dict) + '\n')
        output_dict = {}
    fout.close()

if __name__ == "__main__":
	main()
