# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
import math
from typing import List, Dict, Any

import torch
from torch.utils.data.distributed import DistributedSampler

try:
    from pytext.utils.file_io import PathManager
except ImportError:

    class DummyPathManager:
        def get_local_path(self, path, *args, **kwargs):
            return path

        def open(self, path, *args, **kwargs):
            return open(path, *args, **kwargs)

    PathManager = DummyPathManager()


def maybe_add_title(text, title, use_title, sep_token):
    if use_title:
        return " ".join([title, sep_token, text])
    else:
        return text


class ContiguousDistributedSampler(DistributedSampler):
    def __init__(
        self,
        dataset,
        num_replicas=None,
        rank=None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        num_replicas_per_node: int = 1,
    ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.num_replicas_per_node = num_replicas_per_node

    def __iter__(self):
        indices = list(range(len(self.dataset)))  # type: ignore

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample chunk
        chunk_size = self.num_samples * self.num_replicas_per_node
        node_rank = self.rank // self.num_replicas_per_node
        local_rank = self.rank % self.num_replicas_per_node
        start_idx = node_rank * chunk_size
        indices = indices[start_idx : start_idx + chunk_size]
        if self.shuffle:
            # deterministically shuffle
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch + node_rank)
            shuffle_idx = torch.randperm(
                len(indices), generator=g
            ).tolist()  # type: ignore
            indices = [indices[idx] for idx in shuffle_idx]
        # subsample
        indices = indices[local_rank :: self.num_replicas_per_node]
        assert len(indices) == self.num_samples

        return iter(indices)


class ContiguousDistributedSamplerForTest(DistributedSampler):
    def __iter__(self):
        shard_size = len(self.dataset) // self.num_replicas + 1
        return iter(
            range(
                self.rank * shard_size,
                min((self.rank + 1) * shard_size, len(self.dataset)),
            )
        )


class WrapTransform(torch.nn.Module):
    def __init__(self, transform):
        super().__init__()
        self.transform = transform

    def forward(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        batch: Dict[str, Any] = {"text": texts}
        return self.transform(batch)


class ScriptEncoder(torch.nn.Module):
    # For scripting RobertaEncoder like classes
    def __init__(self, transform, encoder, quantize=False):
        super().__init__()
        self.transform = WrapTransform(transform)
        self.encoder = copy.deepcopy(encoder).cpu()
        if quantize:
            self.encoder = torch.quantization.quantize_dynamic(
                self.encoder, {torch.nn.Linear}, dtype=torch.qint8
            )
        self.cpu()

    def forward(self, texts: List[str]) -> torch.Tensor:
        batch = self.transform(texts)
        return self.encode(batch["token_ids"])

    def encode(self, model_inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(model_inputs)


class ScriptMultiEncoder(torch.nn.Module):
    # For scripting an weighted ensemble of RobertaEncoder like classes
    def __init__(self, transform, encoders, quantize=False, weights=None):
        super().__init__()
        self.transform = WrapTransform(transform)
        self.encoders = torch.nn.ModuleList()
        self.linear = torch.nn.Linear(len(encoders), 1, bias=False, device='cpu')
        if weights is None:
            self.linear.weight.data = torch.ones(
                len(encoders), 1, device="cpu"
            )  # n_enc * 1, by default all ones
        else:
            assert len(weights) == len(encoders)
            self.linear.weight.data = torch.Tensor([weights], device="cpu").T
        for encoder in encoders:
            enc = copy.deepcopy(encoder).cpu()
            if quantize:
                enc = torch.quantization.quantize_dynamic(
                    enc, {torch.nn.Linear}, dtype=torch.qint8
                )
            self.encoders.append(enc)
        if quantize:
            self.linear = torch.quantization.quantize_dynamic(
                self.linear, {torch.nn.Linear}, dtype=torch.qint8
            )
        self.cpu()

    def forward(self, texts: List[str]) -> torch.Tensor:
        batch = self.transform(texts)
        return self.encode(batch["token_ids"])

    def encode(self, model_inputs: torch.Tensor) -> torch.Tensor:
        embeddings_list: List[torch.Tensor] = []
        for i, encoder in enumerate(self.encoders):
            embeddings_list.append(
                self.linear.weight.data[i] * encoder(model_inputs)
            )  # weighted concatenation
        return torch.cat(embeddings_list, dim=1)  # n_enc * d
