# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import hydra
import glob
import os
import pickle
import numpy as np
from dpr_scale.conf.config import MainConfig
from tqdm import tqdm
import concurrent.futures
from dpr_scale.index.quantizer import ProductQuantizer

def load_file(path):
    expert_id = int(path.split("/")[-1].split(".")[0])
    with open(path, "rb") as f:
        data = pickle.load(f)
    return expert_id, data

def save_file(data, path):
    with open(path, mode="wb") as f:
        pickle.dump(data, f, protocol=4)

@hydra.main(config_path="../conf", config_name="config")
def main(cfg: MainConfig):
    os.makedirs(cfg.task.output_dir, exist_ok=True)
    cls_path = os.path.join(os.path.dirname(cfg.task.ctx_embeddings_dir), "cls.pkl")
    if os.path.exists(cls_path):
        quantizer = ProductQuantizer(cfg.cls_dim, cfg.sub_vec_dim, cfg.num_centroids, cfg.iter, "train")
        with open(cls_path, "rb") as f:
            data = pickle.load(f)
            try:
                ctx_cls = data.numpy().astype(np.float32)
            except Exception:
                ctx_cls = data.astype(np.float32)
        print("Loading CLS...")
        print("Running Quantization...")
        quantizer.fit(ctx_cls)
        codes = quantizer.encode(ctx_cls)
        codewords = quantizer.get_centroids()
        # dist_table = (codewords[:, None, :, :] * codewords[:, :, None, :]).sum(-1) # M x K x K
        cls_path = os.path.join(os.path.dirname(cfg.task.ctx_embeddings_dir), f"cls_pq{cfg.sub_vec_dim}.pkl")
        save_file((codes, codewords), cls_path)
    
    print("Loading Index...")
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=100)
    quantizer = ProductQuantizer(cfg.dim, cfg.sub_vec_dim, cfg.num_centroids, cfg.iter, "train")
    input_paths = sorted(glob.glob(os.path.join(cfg.task.ctx_embeddings_dir, "*.pkl")))
    data = []
    for input_path in tqdm(input_paths):
        expert_id, (ctx_id, ctx_weight, ctx_repr) = load_file(input_path)
        data.append((expert_id, (ctx_id, ctx_weight, ctx_repr)))

    print("Running Quantization...")
    data = sorted(data, key=lambda x: -len(x[1][0]))
    print(f"Number of indexes to be compressed: {sum([1 if len(x[1][0]) > cfg.threshold else 0 for x in data])}/{len(data)}")
    for expert_id, (ctx_id, ctx_weight, ctx_repr) in tqdm(data):
        if len(ctx_repr) > cfg.threshold:
            ctx_repr = ctx_repr.numpy().astype(np.float32)
            quantizer.fit(ctx_repr)
            codes = quantizer.encode(ctx_repr)
            codewords = quantizer.get_centroids()
            # dist_table = (codewords[:, None, :, :] * codewords[:, :, None, :]).sum(-1) # M x K x K
            executor.submit(save_file, (ctx_id, codes, codewords), os.path.join(cfg.task.output_dir, f"{expert_id}_pq.pkl"))
        else:
            executor.submit(save_file, (ctx_id, ctx_weight, ctx_repr), os.path.join(cfg.task.output_dir, f"{expert_id}.pkl"))
    
if __name__ == "__main__":
    main()
