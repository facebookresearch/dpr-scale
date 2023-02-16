# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from dataclasses import dataclass, field
from typing import List, Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

defaults = [
    "_self_",
    {"task": "dpr"},
    # Model
    {"task/model": "hf_model"},
    # Transform
    {"task/transform": "hf_transform"},
    # Optim
    {"task/optim": "adamw"},
    # Data
    {"datamodule": "default"},
    # Trainer
    {"trainer": "gpu_1_host"},
    # Trainer callbacks
    {"checkpoint_callback": "default"},
]


@dataclass
class MainConfig:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    task: Any = MISSING
    datamodule: Any = MISSING
    trainer: Any = MISSING
    test_only: bool = False
    checkpoint_callback: Any = MISSING

cs = ConfigStore.instance()

cs.store(name="config", node=MainConfig)
