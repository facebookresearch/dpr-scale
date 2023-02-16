#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Any, Dict, List, Tuple

import hydra
import torch
import torch.nn as nn
from dpr_scale.utils.utils import PathManager, ScriptEncoder

from pytorch_lightning import LightningModule
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook

from torch.optim.lr_scheduler import LambdaLR
from torch.serialization import default_restore_location

# Logic and some code from the original https://github.com/fairinternal/TPR/
# melded with DPRTask.
class DPRDistillTask(LightningModule):
    def __init__(
        self,
        transform,
        model,
        datamodule,
        optim,
        warmup_steps: int = 0,
        fp16_grads: bool = False,
        pretrained_checkpoint_path: str = "",
        k=1,  # k for accuracy@k metric
    ):
        super().__init__()
        # save all the task hyperparams
        # so we can instantiate it much easily later.
        self.save_hyperparameters()
        self.transform_conf = (
            transform.text_transform
            if hasattr(transform, "text_transform")
            else transform
        )
        self.model_conf = model
        self.optim_conf = optim
        self.k = k
        self.loss = nn.MSELoss(reduction="sum")
        self.warmup_steps = warmup_steps
        self.fp16_grads = fp16_grads
        self.pretrained_checkpoint_path = pretrained_checkpoint_path
        self.setup_done = False

    def setup(self, stage: str) -> None:
        # skip building model during test.
        # Otherwise, the state dict will be re-initialized
        if stage == "test" and self.setup_done:
            return
        # resetting call_configure_sharded_model_hook attribute so that we could configure model
        self.call_configure_sharded_model_hook = False

        self.query_encoder = hydra.utils.instantiate(
            self.model_conf,
        )
        if self.pretrained_checkpoint_path:
            checkpoint_dict = torch.load(
                PathManager.open(self.pretrained_checkpoint_path, "rb"),
                map_location=lambda s, l: default_restore_location(s, "cpu"),
            )
            self.load_state_dict(checkpoint_dict["state_dict"])
            print(f"Loaded state dict from {self.pretrained_checkpoint_path}")

        self.setup_done = True

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        This hook will be called before loading state_dict from a checkpoint.
        setup("fit") will built the model before loading state_dict
        """
        self.setup("fit")

    def on_pretrain_routine_start(self) -> None:
        if self.fp16_grads:
            self.trainer.strategy._model.register_comm_hook(None, fp16_compress_hook)

    def _encode_sequence(
        self, token_ids: List[List[torch.IntTensor]], encoder_model: torch.nn.Module
    ) -> List[List[torch.FloatTensor]]:
        encoded_seq = encoder_model(token_ids)  # bs x d
        return encoded_seq

    def sim_score(
        self,
        query_repr: List[List[torch.IntTensor]],
        context_repr: List[List[torch.FloatTensor]],
    ) -> List[List[torch.FloatTensor]]:
        scores = torch.matmul(
            query_repr, torch.transpose(context_repr, 0, 1)
        )  # num_q x num_ctx
        return scores

    def compute_rank_metrics(
        self,
        pred_scores: List[List[torch.FloatTensor]],
        target_labels: List[torch.IntTensor],
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        # Compute total un_normalized avg_ranks, mrr
        _, indices = torch.sort(pred_scores, dim=1, descending=True)
        rank = 0
        mrr = 0.0
        score = 0.0
        for i, idx in enumerate(target_labels):
            in_batch_pos_idx = torch.nonzero(indices[i] == idx, as_tuple=False)
            rank += in_batch_pos_idx.item() + 1
            score += in_batch_pos_idx.item() < self.k
            mrr += 1 / (in_batch_pos_idx.item() + 1)
        return rank, mrr, score

    def encode_queries(
        self, query_ids: List[List[torch.IntTensor]]
    ) -> List[List[torch.FloatTensor]]:
        # generate qry embeddings
        query_repr = self._encode_sequence(query_ids, self.query_encoder)  # bs x d
        return query_repr

    def forward(
        self, query_ids: List[List[torch.IntTensor]]
    ) -> List[List[torch.FloatTensor]]:
        # encode qry
        query_repr = self.encode_queries(query_ids)  # bs x d
        return query_repr

    def configure_optimizers(self):
        self.optimizer = hydra.utils.instantiate(self.optim_conf, self.parameters())
        if self.trainer.max_steps and self.trainer.max_steps > 0:
            training_steps = self.trainer.max_steps
        else:
            steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
            training_steps = steps_per_epoch * self.trainer.max_epochs
        print(
            f"Configured LR scheduler for total {training_steps} training steps, "
            f"with {self.warmup_steps} warmup steps."
        )

        def lr_lambda(current_step: int):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            return max(
                0.0,
                float(training_steps - current_step)
                / float(max(1, training_steps - self.warmup_steps)),
            )

        scheduler = LambdaLR(self.optimizer, lr_lambda)
        scheduler = {
            "scheduler": LambdaLR(self.optimizer, lr_lambda),
            "name": "learning_rate",
            "interval": "step",
            "frequency": 1,
        }
        return [self.optimizer], [scheduler]

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.FloatTensor:
        """
        This receives queries, each with mutliple contexts.
        """
        query_ids = batch["query_ids"]  # bs x tokens
        targets = batch["target_vectors"]

        query_repr = self(query_ids)  # bs x emb_dim

        loss = self.loss(query_repr, targets)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def _eval_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Tuple[
        Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor],
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        # in_batch_eval is enabled automatically
        query_ids = batch["query_ids"]  # bs x tokens
        targets = batch["target_vectors"]

        query_repr = self(query_ids)  # bs x emb_dim
        pred_scores = self.sim_score(query_repr, targets)  # num_q x num_ctx

        loss = self.loss(query_repr, targets)
        in_batch_ctx_indices = torch.arange(targets.shape[0])

        return (
            self.compute_rank_metrics(pred_scores, in_batch_ctx_indices),
            query_repr,
            targets,
            loss,
        )

    def _eval_epoch_end(
        self,
        outputs: Tuple[
            Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor],
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
        ],
        log_prefix="valid",
    ) -> None:

        (
            total_loss,
            total_mrr,
            total_avg_rank,
            total_score,
            total_ctx_count,
            total_count,
        ) = (0, 0, 0, 0, 0, 0)

        for metrics, query_repr, targets, loss in outputs:
            rank, mrr, score = metrics
            total_avg_rank += rank
            total_mrr += mrr
            total_count += query_repr.shape[0]
            total_ctx_count += targets.shape[0]
            total_score += score
            total_loss += loss

        total_loss = total_loss / len(outputs)
        total_ctx_count = total_ctx_count / len(outputs)

        metrics = {
            log_prefix + "_loss": total_loss,
            log_prefix + f"_accuracy@{self.k}": total_score / total_count,
            log_prefix + "_avg_rank": total_avg_rank / total_count,
            log_prefix + "_mrr": total_mrr / total_count,
            log_prefix + "_ctx_count": total_ctx_count,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        return self._eval_step(batch, batch_idx)

    def validation_epoch_end(
        self,
        valid_outputs: Tuple[
            Tuple[torch.FloatTensor, torch.FloatTensor],
            torch.FloatTensor,
            torch.FloatTensor,
        ],
    ) -> None:
        self._eval_epoch_end(valid_outputs) if valid_outputs else None

    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        return self._eval_step(batch, batch_idx)

    def test_epoch_end(
        self,
        test_outputs: Tuple[
            Tuple[torch.FloatTensor, torch.FloatTensor],
            torch.FloatTensor,
            torch.FloatTensor,
        ],
    ) -> None:
        self._eval_epoch_end(test_outputs, "test") if test_outputs else None

    @torch.no_grad()
    def to_torchscript(
        self,
        file_path: str = None,
        method: str = "script",
        example_inputs: str = None,
        **kwargs,
    ) -> Dict[str, Any]:

        mode = self.training
        if method == "script":
            transform = hydra.utils.instantiate(self.transform_conf)
            q_encoder = ScriptEncoder(transform, self.query_encoder)
            q_encoder = torch.jit.script(q_encoder.eval(), **kwargs)
            result = {"q_encoder": q_encoder}
            # Quantize. TODO when PL has better handling link this with the save_quantized
            # flag in ModelCheckpoint
            q_encoder_qt = ScriptEncoder(transform, self.query_encoder, quantize=True)
            q_encoder_qt = torch.jit.script(q_encoder_qt.eval(), **kwargs)
            result["q_encoder_qt"] = q_encoder_qt
        else:
            raise ValueError(
                "The 'method' parameter only supports 'script',"
                f" but value given was: {method}"
            )

        self.train(mode)

        if file_path is not None:
            torch.jit.save(q_encoder, file_path)

        return result
