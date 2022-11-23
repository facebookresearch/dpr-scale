# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

# @manual=//faiss/python:pyfaiss
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
    input_paths = sorted(glob.glob(os.path.join(input_dir, f"{expert_id}.pkl")))
    if len(input_paths) == 0:
        return [], [], []
    for input_path in input_paths:
        data.append(load_file(input_path))
    id_data, weight_data, repr_data = zip(*data)
    id_data = torch.cat(id_data, 0)
    weight_data = torch.cat(weight_data, 0)
    repr_data = torch.cat(repr_data, 0)
    return id_data, weight_data, repr_data
    
def save_file(entry):
    path, output = entry
    with open(path, "wb") as f:
        pickle.dump(output, f, protocol=4)

def main():
    ctx_embeddings_dir = sys.argv[1]
    output_dir = sys.argv[2]
    prune_weight = sys.argv[3]
    ranges = sys.argv[4]
    
    os.makedirs(output_dir, exist_ok=True)
    embedding_out_dir = os.path.join(
        output_dir, f"expert_pruned{prune_weight}")
    os.makedirs(embedding_out_dir, exist_ok=True)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1000)
    range_str = ranges.split("-")
    start, end = int(range_str[0]), int(range_str[1])
    for k in tqdm(range(start, end)):
        ctx_id, ctx_weight, ctx_repr = load_context_expert(ctx_embeddings_dir, k)
        if len(ctx_id) == 0:
            continue
        selected = torch.where(ctx_weight > float(prune_weight))
        ctx_id = ctx_id[selected]
        ctx_weight = ctx_weight[selected]
        ctx_repr = ctx_repr[selected]
        if len(ctx_id) == 0:
            continue
        path = os.path.join(embedding_out_dir, f"{k}.pkl")
        executor.submit(save_file, (path, (ctx_id, ctx_weight, ctx_repr)))
    executor.shutdown()
    

        
if __name__ == "__main__":
    main()
