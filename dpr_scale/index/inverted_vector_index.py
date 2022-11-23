import os
import pickle
import collections
import torch
import time
import glob
import numpy as np
from tqdm import tqdm
import torch_scatter
try:
    from dpr_scale.retriever_ext import scatter as c_scatter
except ImportError:
    raise ImportError(
        'Cannot import scatter module.'
        ' Make sure you have compiled the retriever extension.'
    )
from dpr_scale.index.quantizer import ProductQuantizer

class IVFGPUIndex:
    def __init__(self, portion, corpus_len, ctx_embeddings_dir, expert_parallel=True):
        self.portion = portion
        self.expert_parallel = expert_parallel
        self.cached_experts = {}
        self.ctx_embeddings_dir = ctx_embeddings_dir
        self.load_context_expert(ctx_embeddings_dir)
        self.sum_scores = torch.zeros((1, corpus_len,), dtype=torch.float16).cuda()
        self.max_scores = torch.zeros((corpus_len,), dtype=torch.float16).cuda()
        self.latency = collections.defaultdict(float)

    def search(self, cls_vec, embeddings, weights, topk=100):
        tic = time.perf_counter()
        expert_query_repr, expert_query_id, expert_query_weight = self.encode_query(embeddings, weights)
        toc = time.perf_counter()
        self.latency["encode_time"] += toc - tic
        
        tic = time.perf_counter()
        if len(cls_vec) > 0:
            self.cls_search(cls_vec)
        else:
            if len(self.sum_scores) > len(embeddings):
                self.sum_scores = self.sum_scores[:len(embeddings)]
            elif len(self.sum_scores) < len(embeddings):
                self.sum_scores = self.sum_scores[:1].tile(len(embeddings), 1)
        # token search, reduce-max-and-sum
        self.expert_search(expert_query_repr, 
                           expert_query_weight, 
                           expert_query_id,)
        top_scores, top_ids = self.sort(topk)

        if len(cls_vec) == 0:
            self.sum_scores.fill_(0)
        toc = time.perf_counter()
        self.latency["total_search_time"] += toc - tic
        return top_scores, top_ids    
    
    def sort(self, topk):
        tic = time.perf_counter()
        top_scores, top_ids = self.sum_scores.topk(topk, dim=1)
        toc = time.perf_counter()
        self.latency["sort_time"] += toc - tic
        return top_scores, top_ids
    
    def cls_search(self, query_vec):
        tic = time.perf_counter()
        self.sum_scores = self.compute_similarity(query_vec, self.ctx_cls)
        toc = time.perf_counter()
        self.latency["token_retrieval_time"] += toc - tic

    def encode_query(self, sparse_q_reprs, sparse_q_weights):
        expert_query_repr = collections.defaultdict(list)
        expert_query_id = collections.defaultdict(list)
        expert_query_weight = collections.defaultdict(list)
        for i, (sparse_q_repr, sparse_q_weight) in enumerate(zip(sparse_q_reprs, sparse_q_weights)):
            for k, q_reprs in sparse_q_repr.items():
                q_weights = sparse_q_weight[k]
                for q_weight, q_repr in zip(q_weights, q_reprs):
                    expert_query_weight[k].append(q_weight)
                    expert_query_repr[k].append(q_repr)
                    expert_query_id[k].append(i)
        for k in expert_query_repr.keys():
            query_repr = torch.stack(expert_query_repr[k], 0)
            query_weight = torch.stack(expert_query_weight[k], 0)
            expert_query_repr[k] = query_repr
            expert_query_weight[k] = query_weight
        return expert_query_repr, expert_query_id, expert_query_weight
    
    def get_experts(self, expert_query_repr, expert_query_id):
        all_batch_ids = []
        all_q_repr, all_q_lens = [], []
        all_ctx_repr, all_ctx_lens, all_ctx_ids = [], [], []
        for k in expert_query_repr.keys():
            if k in self.cached_experts:
                ctx_id, ctx_repr = self.cached_experts[k]
                if ctx_id.device == torch.device('cpu'):
                    ctx_id, ctx_repr = ctx_id.cuda().to(torch.int64), ctx_repr.cuda().to(torch.float16)
            else:
                ctx_id, ctx_repr = self.load_expert_from_disk(k)
                if len(ctx_id) == 0:
                    continue
            all_batch_ids.append(expert_query_id[k])
            all_q_repr.append(expert_query_repr[k])
            all_q_lens.append(len(expert_query_repr[k]))
            all_ctx_repr.append(ctx_repr)
            all_ctx_lens.append(len(ctx_repr))
            all_ctx_ids.append(ctx_id)
        if len(all_q_repr) > 0:
            all_q_repr = torch.cat(all_q_repr)
            all_ctx_repr = torch.cat(all_ctx_repr)
        return all_batch_ids, all_q_repr, all_q_lens, all_ctx_repr, all_ctx_lens, all_ctx_ids
    
    def expert_search(self, expert_query_repr, expert_query_weight, expert_query_id):
        tic = time.perf_counter()
        all_batch_ids, all_q_repr, all_q_lens, all_ctx_repr, all_ctx_lens, all_ctx_ids = self.get_experts(expert_query_repr, expert_query_id)
        toc = time.perf_counter()
        self.latency["cpu2gpu_time"] += toc - tic
        if len(all_q_repr) > 0:
            # compute similarity
            tic = time.perf_counter()
            all_batch_scores = self.compute_similarity(all_q_repr, all_ctx_repr)
            toc = time.perf_counter()
            self.latency["token_retrieval_time"] += toc - tic
            # scatter ops
            tic = time.perf_counter()
            q_start, ctx_start = 0, 0
            for i, (batch_ids, q_len, ctx_len) in enumerate(zip(all_batch_ids, all_q_lens, all_ctx_lens)):
                q_end, ctx_end = q_start + q_len, ctx_start + ctx_len
                batch_scores = all_batch_scores[q_start:q_end, ctx_start:ctx_end]
                ctx_id = all_ctx_ids[i]
                for batch_id, scores in zip(batch_ids, batch_scores):
                    torch_scatter.scatter_max(src=scores, index=ctx_id, out=self.max_scores, dim=-1)
                    self.sum_scores[batch_id] += self.max_scores
                    self.max_scores.fill_(0)
                q_start = q_end
                ctx_start = ctx_end
            toc = time.perf_counter()
            self.latency["scatter_time"] += toc - tic

    def compute_similarity(self, q_repr, ctx_repr):
        return torch.matmul(q_repr, ctx_repr.T)
        
    def load_context_expert(self, input_dir):
        print("Loading Index...")
        cls_path = os.path.join(os.path.dirname(input_dir), "cls.pkl")
        gpu_memory = 0
        if os.path.exists(cls_path):
            with open(cls_path, "rb") as f:
                data = pickle.load(f)
                try:
                    self.ctx_cls = data.cuda().to(torch.float16)
                except Exception:
                    self.ctx_cls = torch.from_numpy(data).cuda().to(torch.float16)
            gpu_memory += self.ctx_cls.nelement() * self.ctx_cls.element_size()

        cache = []
        input_paths = sorted(glob.glob(os.path.join(input_dir, "*.pkl")))
        for input_path in tqdm(input_paths):
            expert_id = int(input_path.split("/")[-1].split(".")[0])
            id_data, _, repr_data = self.load_file(input_path)
            cache.append((expert_id, id_data, repr_data))
        # sort the index from large to small
        cache = sorted(cache, key=lambda x: -len(x[2]))
        gpu_end = int(len(cache) * self.portion)
        cpu_end = int(len(cache))

        # GPU portion
        for k, id_data, repr_data in cache[:gpu_end]:
            id_data, repr_data = id_data.cuda().to(torch.int64), repr_data.cuda().to(torch.float16)
            gpu_memory += id_data.nelement() * id_data.element_size() + repr_data.nelement() * repr_data.element_size()
            self.cached_experts[k] = (id_data, repr_data)
        print(f"GPU index usage: {gpu_memory/(1024**3):.4f} GB")
        
        cpu_memory = 0
        # CPU portion
        for k, id_data, repr_data in cache[gpu_end:cpu_end]:
            id_data, repr_data = id_data.to(torch.int64), repr_data.to(torch.float16)
            cpu_memory += id_data.nelement() * id_data.element_size() + repr_data.nelement() * repr_data.element_size()
            self.cached_experts[k] = (id_data, repr_data)
        print(f"CPU index usage: {cpu_memory/(1024**3):.4f} GB")
        print(len(self.cached_experts))     
        # The rest will be stored in disk and loaded into memory on the fly
    
    def load_expert_from_disk(self, expert_id):
        input_path = os.path.join(self.ctx_embeddings_dir, f"{expert_id}.pkl")
        if os.path.exists(input_path):
            data = self.load_file(input_path)
        else:
            return [], []
        id_data, weight_data, repr_data = data
        return id_data.to(torch.int64).cuda(), repr_data.to(torch.float16).cuda()
    
    def load_file(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data

class IVFPQGPUIndex(IVFGPUIndex):
    def __init__(self, *args, cls_dim=128, dim=32, sub_vec_dim=4, num_centroids=256, **kwargs):
        self.cached_pq_experts = {}
        self.cls_dim = cls_dim
        self.dim = dim
        self.sub_vec_dim = sub_vec_dim
        self.num_centroids = num_centroids
        super().__init__(*args, **kwargs)
    
    def get_experts(self, expert_query_repr, expert_query_id):
        all_batch_ids = []
        all_q_repr, all_q_lens = [], []
        all_ctx_repr, all_ctx_lens, all_ctx_ids = [], [], []
        for k in expert_query_repr.keys():
            if k in self.cached_pq_experts:
                ctx_id, ctx_codes, ctx_codewords = self.cached_pq_experts[k]
                ctx_repr = self.decode(ctx_codes.to(torch.int64), ctx_codewords, self.dim)
            elif k in self.cached_experts:
                ctx_id, ctx_repr = self.cached_experts[k]
                if ctx_id.device == torch.device('cpu'):
                    ctx_id, ctx_repr = ctx_id.cuda().to(torch.int64), ctx_repr.cuda().to(torch.float16)
            else:
                ctx_id, ctx_repr = self.load_expert_from_disk(k)
                if len(ctx_id) == 0:
                    continue
            all_batch_ids.append(expert_query_id[k])
            all_q_repr.append(expert_query_repr[k])
            all_q_lens.append(len(expert_query_repr[k]))
            all_ctx_repr.append(ctx_repr)
            all_ctx_lens.append(len(ctx_repr))
            all_ctx_ids.append(ctx_id)
        if len(all_q_repr) > 0:
            all_q_repr = torch.cat(all_q_repr)
            all_ctx_repr = torch.cat(all_ctx_repr)
        return all_batch_ids, all_q_repr, all_q_lens, all_ctx_repr, all_ctx_lens, all_ctx_ids
        
    def load_context_expert(self, input_dir):
        print("Loading Index...")
        cls_path = os.path.join(os.path.dirname(input_dir), f"cls_pq{self.sub_vec_dim}.pkl")
        gpu_memory = 0
        if os.path.exists(cls_path):
            with open(cls_path, "rb") as f:
                data = pickle.load(f)
                code_data, codeword_data = data
                self.ctx_cls = self.decode(torch.from_numpy(code_data).cuda().to(torch.int64), torch.from_numpy(codeword_data).cuda().to(torch.float16), self.cls_dim)
            gpu_memory += self.ctx_cls.nelement() * self.ctx_cls.element_size()

        cache = []    
        pq_cache = []
        input_paths = sorted(glob.glob(os.path.join(input_dir, "*.pkl")))
        for input_path in tqdm(input_paths):
            expert_id = input_path.split("/")[-1].split(".")[0]
            if "pq" in expert_id:
                expert_id = expert_id.split("_")[0]
                id_data, code_data, codeword_data = self.load_file(input_path)
                pq_cache.append((int(expert_id), id_data, code_data, codeword_data))
            else:
                id_data, weights, repr_data = self.load_file(input_path)
                cache.append((int(expert_id), id_data, repr_data))

        # sort the index from large to small
        cache = sorted(cache, key=lambda x: -len(x[2]))
        # GPU portion
        for k, id_data, repr_data in cache:
            id_data, repr_data = id_data.cuda().to(torch.int64), repr_data.cuda().to(torch.float16)
            gpu_memory += id_data.nelement() * id_data.element_size() + repr_data.nelement() * repr_data.element_size()
            self.cached_experts[k] = (id_data, repr_data)
        
        for k, id_data, code_data, codeword_data in pq_cache:
            id_data, code_data, codeword_data = id_data.to(torch.int64).cuda(), torch.ByteTensor(code_data).cuda(), torch.HalfTensor(codeword_data).cuda()
            gpu_memory += id_data.nelement() * id_data.element_size() + code_data.nelement() * code_data.element_size() + codeword_data.nelement() * codeword_data.element_size()
            self.cached_pq_experts[k] = (id_data, code_data, codeword_data)
        print(f"GPU index usage: {gpu_memory/(1024**3):.4f} GB")
        # The rest will be stored in disk and loaded into memory on the fly
    
    def decode(self, codes, centroids, dim):
        codes = codes.permute(1, 0).unsqueeze(-1).tile(1, 1, self.sub_vec_dim)
        vecs = torch.gather(centroids, 1, codes).permute(1, 0, 2).reshape((-1, dim))
        return vecs

        
class IVFCPUIndex:
    def __init__(self, portion, corpus_len, ctx_embeddings_dir):
        self.portion = portion
        self.cached_experts = {}
        self.ctx_embeddings_dir = ctx_embeddings_dir
        self.load_context_expert(ctx_embeddings_dir)
        self.sum_scores = np.zeros((1, corpus_len,), dtype=np.float32)
        self.latency = collections.defaultdict(float)

    def search(self, cls_vec, embeddings, weights, topk=100):
        tic = time.perf_counter()
        cls_vec = cls_vec.cpu().numpy().astype(np.float32)
        expert_query_repr, expert_query_id, expert_query_weight = self.encode_query(embeddings, weights)
        toc = time.perf_counter()
        self.latency["encode_time"] += toc - tic

        tic = time.perf_counter()
        if len(cls_vec) > 0:
            self.cls_search(cls_vec)
        else:
            if len(self.sum_scores) > len(embeddings):
                self.sum_scores = self.sum_scores[:len(embeddings)]
            elif len(self.sum_scores) < len(embeddings):
                self.sum_scores = self.sum_scores[:1].tile((len(embeddings), 1))
        # token search, reduce-max-and-sum
        self.expert_search(expert_query_repr, 
                           expert_query_weight, 
                           expert_query_id,)
        top_scores, top_ids = self.sort(topk)
        self.sum_scores.fill(0)
        toc = time.perf_counter()
        self.latency["search_time"] += toc - tic
        return top_scores, top_ids    
    
    def cls_search(self, query_vec):
        tic = time.perf_counter()
        self.sum_scores = self.compute_similarity(query_vec, self.ctx_cls)
        toc = time.perf_counter()
        self.latency["token_retrieval_time"] += toc - tic
    
    def sort(self, topk):
        tic = time.perf_counter()
        top_ids = np.argpartition(self.sum_scores, -topk, axis=1)[:, -topk:] # linear time partition but shuffled
        top_scores = np.take_along_axis(self.sum_scores, top_ids, axis=1)
        top_subset_ids = np.argsort(-1.*top_scores, axis=1) # sort the top-k list
        top_scores = np.take_along_axis(top_scores, top_subset_ids, axis=1)
        top_ids = np.take_along_axis(top_ids, top_subset_ids, axis=1)
        toc = time.perf_counter()
        self.latency["sort_time"] += toc - tic
        return top_scores, top_ids
    
    def encode_query(self, sparse_q_reprs, sparse_q_weights):
        expert_query_repr = collections.defaultdict(list)
        expert_query_id = collections.defaultdict(list)
        expert_query_weight = collections.defaultdict(list)
        for i, (sparse_q_repr, sparse_q_weight) in enumerate(zip(sparse_q_reprs, sparse_q_weights)):
            for k, q_reprs in sparse_q_repr.items():
                q_weights = sparse_q_weight[k]
                for q_weight, q_repr in zip(q_weights, q_reprs):
                    expert_query_weight[k].append(q_weight)
                    expert_query_repr[k].append(q_repr)
                    expert_query_id[k].append(i)
        
        for k in expert_query_repr.keys():
            query_repr = torch.stack(expert_query_repr[k], 0)
            query_weight = torch.stack(expert_query_weight[k], 0)
            expert_query_repr[k] = query_repr.cpu().numpy().astype(np.float32)
            expert_query_weight[k] = query_weight.cpu().numpy().astype(np.float32)
        
        return expert_query_repr, expert_query_id, expert_query_weight
    
    def get_experts(self, k):
        if k in self.cached_experts:
            ctx_id, ctx_repr = self.cached_experts[k]
        else:
            ctx_id, ctx_repr = self.load_expert_from_disk(k)
        return ctx_id, ctx_repr

    def expert_search(self, expert_query_repr, expert_query_weight, expert_query_id):
        for k in expert_query_repr.keys():
            tic = time.perf_counter()
            ctx_id, ctx_repr = self.get_experts(k)
            toc = time.perf_counter()
            self.latency["disk2cpu"] += toc - tic
            if len(ctx_id) == 0:
                continue
            q_repr = expert_query_repr[k]
            batch_ids = expert_query_id[k]
            
            # compute similarity
            tic = time.perf_counter()
            batch_scores = self.compute_similarity(q_repr, ctx_repr)
            toc = time.perf_counter()
            self.latency["token_retrieval_time"] += toc - tic

            tic = time.perf_counter()
            for batch_id, scores in zip(batch_ids, batch_scores):
                max_scores = np.empty(self.sum_scores.shape[1], dtype=np.float32)
                c_scatter.scatter_max(scores, ctx_id, max_scores)
                ctx_index = np.unique(ctx_id)
                c_scatter.scatter_add(max_scores, ctx_index, self.sum_scores[batch_id])
            toc = time.perf_counter()
            self.latency["scatter_time"] += toc - tic
            
    def compute_similarity(self, q_repr, ctx_repr):
        return np.matmul(q_repr, ctx_repr.T)
        
    def load_context_expert(self, input_dir):
        print("Loading Index...")
        cls_path = os.path.join(os.path.dirname(input_dir), "cls.pkl")
        memory = 0
        if os.path.exists(cls_path):
            with open(cls_path, "rb") as f:
                self.ctx_cls = pickle.load(f)
            memory += self.ctx_cls.nelement() * self.ctx_cls.element_size()
            self.ctx_cls = self.ctx_cls.numpy().astype(np.float32)

        cache = []    
        input_paths = sorted(glob.glob(os.path.join(input_dir, "*.pkl")))
        for input_path in tqdm(input_paths):
            expert_id = int(input_path.split("/")[-1].split(".")[0])
            id_data, _, repr_data = self.load_file(input_path)
            cache.append((expert_id, id_data, repr_data))

        # sort the index from large to small
        cache = sorted(cache, key=lambda x: -len(x[2]))
        cpu_end = int(len(cache) * self.portion)
        # CPU portion
        for k, id_data, repr_data in cache[:cpu_end]:
            memory += id_data.nelement() * id_data.element_size() + repr_data.nelement() * repr_data.element_size()
            self.cached_experts[k] = (id_data.numpy().astype(np.int64), repr_data.numpy().astype(np.float32))
        print(f"CPU index usage: {memory/(1024**3):.4f} GB")        
        # The rest will be stored in disk and loaded into memory on the fly
    
    def load_expert_from_disk(self, expert_id):
        input_path = os.path.join(self.ctx_embeddings_dir, f"{expert_id}.pkl")
        if os.path.exists(input_path):
            data = self.load_file(input_path)
        else:
            return [], []
        id_data, _, repr_data = data
        return id_data.numpy().astype(np.int64), repr_data.numpy().astype(np.float32)
    
    def load_file(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data


class IVFPQCPUIndex(IVFCPUIndex):
    def __init__(self, *args, dim=32, sub_vec_dim=4, num_centroids=256, **kwargs):
        self.cached_pq_experts = {}
        self.dim = dim
        self.sub_vec_dim = sub_vec_dim
        self.num_centroids = num_centroids
        super().__init__(*args, **kwargs)
    
    def get_experts(self, k):
        if k in self.cached_pq_experts:
            ctx_id, ctx_codes, quantizer = self.cached_pq_experts[k]
            ctx_repr = quantizer.decode(ctx_codes)
        else:
            ctx_id, ctx_repr = self.load_expert_from_disk(k)
        return ctx_id, ctx_repr
        
    def load_context_expert(self, input_dir):
        print("Loading Index...")
        cls_path = os.path.join(os.path.dirname(input_dir), f"cls_pq{self.sub_vec_dim}.pkl")
        memory = 0
        if os.path.exists(cls_path):
            with open(cls_path, "rb") as f:
                data = pickle.load(f)
                code_data, codeword_data = data
                self.ctx_cls = self.decode(torch.from_numpy(code_data).to(torch.int64), torch.from_numpy(codeword_data).to(torch.float32), self.cls_dim)
            memory += self.ctx_cls.nelement() * self.ctx_cls.element_size()

        cache = []    
        pq_cache = []
        input_paths = sorted(glob.glob(os.path.join(input_dir, "*.pkl")))
        for input_path in tqdm(input_paths):
            expert_id = input_path.split("/")[-1].split(".")[0]
            if "pq" in expert_id:
                expert_id = expert_id.split("_")[0]
                id_data, code_data, codeword_data = self.load_file(input_path)
                pq_cache.append((int(expert_id), id_data, code_data, codeword_data))
            else:
                id_data, _, repr_data = self.load_file(input_path)
                cache.append((int(expert_id), id_data, repr_data))

        # sort the index from large to small
        cache = sorted(cache, key=lambda x: -len(x[2]))
        # CPU portion
        for k, id_data, repr_data in cache:
            memory += id_data.nelement() * id_data.element_size() + repr_data.nelement() * repr_data.element_size()
            self.cached_experts[k] = (id_data.numpy().astype(np.int64), repr_data.numpy().astype(np.float32))
        # CPU portion
        for k, id_data, code_data, codeword_data in pq_cache:
            id_data, code_data, codeword_data = id_data.numpy().astype(np.int64), code_data, codeword_data
            memory += id_data.size * id_data.itemsize + code_data.size * code_data.itemsize + codeword_data.size * codeword_data.itemsize
            quantizer = ProductQuantizer(self.dim, self.sub_vec_dim, self.num_centroids, mode="test")
            quantizer.set_centroids(codeword_data)
            self.cached_pq_experts[k] = (id_data, code_data, quantizer)
        print(f"CPU index usage: {memory/(1024**3):.4f} GB")        
        # The rest will be stored in disk and loaded into memory on the fly
