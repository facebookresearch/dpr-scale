# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import hydra
from dpr_scale.conf.config import MainConfig
from omegaconf import open_dict
from pytorch_lightning.trainer import Trainer


@hydra.main(config_path="conf", config_name="config")
def main(cfg: MainConfig):
    # Temp patch for datamodule refactoring
    cfg.task.datamodule = None
    cfg.task._target_ = "dpr_scale.task.dpr_eval_task.GenerateEmbeddingsTask"  # hack
    task = hydra.utils.instantiate(cfg.task, _recursive_=False)
    transform = hydra.utils.instantiate(cfg.task.transform)
    datamodule = hydra.utils.instantiate(cfg.datamodule, transform=transform)

    # trainer.fit does some setup, so we need to call it even though no training is done
    with open_dict(cfg):
        cfg.trainer.limit_train_batches = 0
        if "plugins" in cfg.trainer:
            cfg.trainer.pop("plugins")  # remove ddp_sharded, since it breaks during loading
    print(cfg)
    trainer = Trainer(**cfg.trainer)
    trainer.fit(task, datamodule=datamodule)
    trainer.test(task, datamodule=datamodule)


if __name__ == "__main__":
    main()
