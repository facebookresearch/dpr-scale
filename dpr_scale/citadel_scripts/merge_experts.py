# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import glob
import torch
import os
import sys
import pickle
from tqdm import tqdm
import concurrent.futures

def load_context_expert(input_dir, expert_id):
    def load_file(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data
    data = []                
    input_paths = sorted(glob.glob(os.path.join(input_dir, "expert_*", f"{expert_id}.pkl")))
    if len(input_paths) == 0:
        return [], [], []
    for input_path in input_paths:
        data.append(load_file(input_path))
    id_data, weight_data, repr_data = zip(*data)
    id_data = torch.cat(id_data, 0)
    weight_data = torch.cat(weight_data, 0)
    repr_data = torch.cat(repr_data, 0)
    return id_data, weight_data, repr_data

def load_context_cls(input_dir):
    def load_file(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data
    input_cls_paths = sorted(glob.glob(os.path.join(input_dir, "cls_*.pkl")))
    cls_embeddings = []
    if len(input_cls_paths) > 0:
        for input_cls_path in tqdm(input_cls_paths):
            cls_embeddings.append(load_file(input_cls_path))
        cls_embeddings = torch.cat(cls_embeddings, 0)
    return cls_embeddings

def save_file(entry):
    path, output = entry
    with open(path, "wb") as f:
        pickle.dump(output, f, protocol=4)

def main():
    os.makedirs(sys.argv[1], exist_ok=True)
    embedding_out_path = os.path.join(
        sys.argv[1], "cls.pkl")
    if not os.path.exists(embedding_out_path):
        ctx_cls_embeddings = load_context_cls(sys.argv[2])
        if len(ctx_cls_embeddings) > 0:
            save_file((embedding_out_path, ctx_cls_embeddings))

    embedding_out_dir = os.path.join(
        sys.argv[1], "expert")
    os.makedirs(embedding_out_dir, exist_ok=True)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1000)
    range_str = sys.argv[3].split("-")
    start, end = int(range_str[0]), int(range_str[1])
    for k in tqdm(range(start, end)):
        ctx_id, ctx_weight, ctx_repr = load_context_expert(sys.argv[2], k)
        if len(ctx_id) == 0:
            continue
        path = os.path.join(embedding_out_dir, f"{k}.pkl")
        executor.submit(save_file, (path, (ctx_id, ctx_weight, ctx_repr)))
    executor.shutdown()
    

        
if __name__ == "__main__":
    main()