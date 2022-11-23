# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import hydra
import glob
import json
import os
from dpr_scale.conf.config import MainConfig
from omegaconf import open_dict
from pytorch_lightning.trainer import Trainer

@hydra.main(config_path="../conf", config_name="config")
def main(cfg: MainConfig):
    # Temp patch for datamodule refactoring
    cfg.task.datamodule = None
    cfg.task._target_ = (
        "dpr_scale.task.citadel_retrieval_task.CITADELRetrievalTask"  # hack
    )
    # trainer.fit does some setup, so we need to call it even though no training is done
    with open_dict(cfg):
        cfg.trainer.limit_train_batches = 0
        if "plugins" in cfg.trainer:
            cfg.trainer.pop(
                "plugins"
            )  # remove ddp_sharded, because it breaks during loading

    print(cfg)

    task = hydra.utils.instantiate(cfg.task, _recursive_=False)
    transform = hydra.utils.instantiate(cfg.task.transform)
    datamodule = hydra.utils.instantiate(cfg.datamodule, transform=transform)

    trainer = Trainer(**cfg.trainer)
    trainer.test(task, datamodule=datamodule)
    
    if cfg.datamodule.trec_format:
        input_paths = sorted(glob.glob(os.path.join(cfg.task.output_path, "retrieval_*.trec")))
        result = []
        for input_path in input_paths:
            with open(input_path) as f:
                data = f.readlines()
                result.extend(data)
        
        with open(os.path.join(cfg.task.output_path, "retrieval.trec"), "w") as f:
            f.writelines(result)
    else:
        input_paths = sorted(glob.glob(os.path.join(cfg.task.output_path, "retrieval_*.json")))
        result = []
        for input_path in input_paths:
            with open(input_path) as f:
                data = json.load(f)
                result.extend(data)
        
        with open(os.path.join(cfg.task.output_path, "retrieval.json"), "w") as g:
            g.write(json.dumps(result, indent=4))
            g.write("\n")

if __name__ == "__main__":
    main()
