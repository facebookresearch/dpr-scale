#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from pytorch_lightning.strategies import DDPShardedStrategy, DDPStrategy
from dpr_scale.task.dpr_task import DenseRetrieverTask

class MultiVecRetrieverTask(DenseRetrieverTask):
    def __init__(
        self,
        add_cls: bool = False,
        query_topk: int = 1,
        context_topk: int = 1,
        query_expert_load_loss_coef: float = 0,
        context_expert_load_loss_coef: float = 0,
        query_router_marg_load_loss_coef: float = 0,
        context_router_marg_load_loss_coef: float = 0,
        cross_batch: bool = True,
        in_batch: bool = True,
        query_pool: str = "sum",
        anneal_factor: float = 0.0,
        teacher_coef: float = 0.0,
        tau: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.query_kwargs = dict(
                topk=query_topk, 
                add_cls=add_cls,
        )
        self.context_kwargs = dict(
                topk=context_topk,
                add_cls=add_cls,
        )

        self.query_expert_load_loss_coef = query_expert_load_loss_coef
        self.context_expert_load_loss_coef = context_expert_load_loss_coef
        self.query_router_marg_load_loss_coef = query_router_marg_load_loss_coef 
        self.context_router_marg_load_loss_coef = context_router_marg_load_loss_coef 
        self.cross_batch = cross_batch
        self.in_batch = in_batch
        self.query_pool = query_pool
        self.epoch = 0
        self.anneal_factor = anneal_factor
        self.teacher_coef = teacher_coef
        self.tau = tau
    
    def anneal_func(self, loss):
        coef = min(1, (self.epoch/self.trainer.max_epochs)**self.anneal_factor)
        loss = coef * loss
        self.log("train_anneal_coef", coef)
        return loss
            
    def _encode_sequence(self, token_ids, encoder_model, **kwargs):
        encoded_seq = encoder_model(token_ids, **kwargs)
        return encoded_seq

    def encode_queries(self, query_ids):
        query_repr = self._encode_sequence(query_ids, self.query_encoder, **self.query_kwargs)
        return query_repr

    def encode_contexts(self, contexts_ids):
        contexts_repr = self._encode_sequence(
            contexts_ids, self.context_encoder, **self.context_kwargs
        )
        return contexts_repr

    def pad(self, tensor_list):
        # pad the token embedding sequences from different gpus to the same length
        all_lens = [tensor.size(1) for tensor in tensor_list] # assume the first dimension is batch size
        max_len = max(all_lens)
        padded_tensor_list = []
        for tensor in tensor_list:
            size = [tensor.shape[0]] + [max_len - tensor.size(1)] + list(tensor.shape[2:])
            tensor = torch.cat([tensor, tensor.new_full(size, 0)], dim=1)
            padded_tensor_list.append(tensor)
        return padded_tensor_list
    
    def evenly_divisible_all_gather(self, tensor_list):
        # all_gather func for dynamic-shape tensors: https://github.com/pytorch/ignite/issues/1569
        gathered_tensor_list = []
        for tensor in tensor_list:
            if len(tensor.shape) <= 1:
                gathered_tensor_list.append(self.all_gather(tensor))
            else:
                length = tensor.shape[1]
                all_lens = self.all_gather(length)
                max_len = max(all_lens).item()
                if length < max_len:
                    size = [tensor.shape[0]] + [max_len - length] + list(tensor.shape[2:])
                    tensor = torch.cat([tensor, tensor.new_full(size, 0)], dim=1)
                # all gather across all processes
                tensor = self.all_gather(tensor)
                gathered_tensor_list.append(tensor)
        return gathered_tensor_list

    def distributed_gather(self, query_repr, context_repr, mask, pos_ctx_indices, teacher_scores):
        query_keys = [key for key in query_repr.keys()]
        context_keys = [key for key in context_repr.keys()]
        query_tensors_to_send = [val.detach() for val in query_repr.values()]
        context_tensors_to_send = [val.detach() for val in context_repr.values()]
        # assumes all nodes have same number of contexts
        gathered_tensors = self.evenly_divisible_all_gather(
            (*query_tensors_to_send,
                *context_tensors_to_send,
            pos_ctx_indices, mask, teacher_scores)
        )
        gathered_query_tensors = gathered_tensors[:len(query_tensors_to_send)]
        gathered_context_tensors = gathered_tensors[len(query_tensors_to_send):len(query_tensors_to_send)+len(context_tensors_to_send)]
        all_labels, all_mask, all_teacher_scores = gathered_tensors[-3], gathered_tensors[-2], gathered_tensors[-1]

        context_offset = 0
        query_tensor_list = [[] for _ in query_keys]
        context_tensor_list = [[] for _ in context_keys]

        for i in range(all_labels.size(0)):
            all_labels[i] += context_offset
            query_iterable = zip(query_keys, gathered_query_tensors) if i != self.global_rank else query_repr.items()
            context_iterable = zip(context_keys, gathered_context_tensors) if i != self.global_rank else context_repr.items()

            for j, (_, tensor) in enumerate(query_iterable):
                t = tensor[i] if i != self.global_rank else tensor
                query_tensor_list[j].append(t)
            
            for j, (_, tensor) in enumerate(context_iterable):
                t = tensor[i] if i != self.global_rank else tensor
                context_tensor_list[j].append(t)
            context_offset += t.size(0)
            
        query_repr = {key: torch.cat(self.pad(value), dim=0) for key, value in zip(query_keys, query_tensor_list)}
        context_repr = {key: torch.cat(self.pad(value), dim=0) for key, value in zip(context_keys, context_tensor_list)}
        pos_ctx_indices = torch.flatten(all_labels)  # total_query
        mask = torch.flatten(all_mask)  # total_ctx
        teacher_scores = all_teacher_scores.view(-1, all_teacher_scores.shape[-1])
        return query_repr, context_repr, mask, pos_ctx_indices, teacher_scores
    
    def sim_score(self, query_repr, context_repr, mask=None, pairwise=False):
        if pairwise:
            multiplier = context_repr.shape[0] // query_repr.shape[0] # num_ctx
            query_repr = query_repr.unsqueeze(1) # B x 1 x D
            mask = mask.view(-1, multiplier) # B x num_ctx
            context_repr = context_repr.view(-1, multiplier, context_repr.shape[1]) # B x num_ctx x D
            scores = (query_repr * context_repr).sum(-1)
            if mask is not None:
                scores[mask] = float("-inf")
        else:
            scores = torch.matmul(
                query_repr, torch.transpose(context_repr, 0, 1)
            )  # num_q x num_ctx
            if mask is not None:
                # mask is size num_ctx
                scores[mask.repeat(scores.size(0), 1)] = float("-inf")
        return scores
    
    def colbert_score(self, query_repr, context_repr, pairwise):
        token_dim = query_repr["expert_repr"].size(-1)
        if pairwise:  # for triplet loss or cross-encoder distillation
            multiplier = context_repr["expert_repr"].shape[0] // query_repr["expert_repr"].shape[0] # number of hard negatives
            query_vec = query_repr["expert_repr"].unsqueeze(1).tile(1, multiplier, 1, 1) # Q x M x LQ x dim
            context_vec = context_repr["expert_repr"].view(-1, multiplier, context_repr["expert_repr"].shape[1], token_dim) # Q x M x LD x dim
            scores = torch.einsum("ijqc,ijdc->iqjd", query_vec, context_vec) # Q x LQ x M x LD
        else:  # inbatch negative
            scores = torch.matmul(query_repr["expert_repr"].view(-1, token_dim), context_repr["expert_repr"].view(-1, token_dim).transpose(0, 1))
            scores = scores.view(
                *query_repr["expert_repr"].shape[:2], *context_repr["expert_repr"].shape[:2]) # Q * LQ * D * LD
        return scores

    def coil_score(self, scores, query_repr, context_repr, pairwise):
        # See https://github.com/luyug/COIL/blob/6d15679f9ddb8f29c814d19e674333511c45feb3/modeling.py#L173
        if pairwise:  # triplet
            multiplier = context_repr["expert_repr"].shape[0] // query_repr["expert_repr"].shape[0] # number of hard negatives
            query_expert_ids = query_repr["expert_ids"].unsqueeze(1).unsqueeze(-1).tile(1, multiplier, 1, 1)
            context_expert_ids = context_repr["expert_ids"].view(-1, multiplier, 1, context_repr["expert_ids"].shape[1])
            exact_match = torch.permute((query_expert_ids == context_expert_ids), (0, 2, 1, 3))  # Q * LQ * M * LD
        else:  # inbatch
            exact_match = query_repr["expert_ids"].unsqueeze(2).unsqueeze(3) == context_repr["expert_ids"].unsqueeze(0).unsqueeze(1)  # Q * LQ * D * LD
        
        exact_match = exact_match.to(query_repr["expert_repr"].dtype)
        
        if "expert_weights" in query_repr:
            if pairwise:  # triplet
                query_expert_weights = query_repr["expert_weights"].unsqueeze(1).unsqueeze(-1).tile(1, multiplier, 1, 1)
                context_expert_weights = context_repr["expert_weights"].view(-1, multiplier, 1, context_repr["expert_weights"].shape[1])
                exact_match_weights = torch.permute((query_expert_weights * context_expert_weights), (0, 2, 1, 3))  # Q * LQ * M * LD
            else:  # inbatch
                exact_match_weights = query_repr["expert_weights"].unsqueeze(2).unsqueeze(3) * context_repr["expert_weights"].unsqueeze(0).unsqueeze(1)  # Q * LQ * D * LD    
            exact_match *= exact_match_weights
        scores = scores * exact_match
        return scores
    
    def citadel_score(self, scores, query_repr, context_repr, pairwise):
        if pairwise:  # triplet
            multiplier = context_repr["expert_repr"].shape[0] // query_repr["expert_repr"].shape[0] # number of hard negatives
            query_expert_ids = query_repr["expert_ids"].unsqueeze(1).unsqueeze(-1).unsqueeze(-1).tile(1, multiplier, 1, 1, 1, 1) # Q x M x LQ x KQ x 1 x 1
            context_expert_ids = context_repr["expert_ids"].view(-1, multiplier, 1, 1, context_repr["expert_ids"].shape[1], context_repr["expert_ids"].shape[2]) # Q x M x 1 x 1 x LD x KD
            exact_match = (query_expert_ids == context_expert_ids).permute((0, 2, 3, 1, 4, 5))  # Q * LQ * KQ * M * LD * KD
        else:  # inbatch
            exact_match = (query_repr["expert_ids"].unsqueeze(3).unsqueeze(4).unsqueeze(5) == context_repr["expert_ids"].unsqueeze(0).unsqueeze(1).unsqueeze(2))  # Q * LQ * KQ * D * LD * KD
        
        if "expert_weights" in query_repr:
            if pairwise:  # triplet
                query_expert_weights = query_repr["expert_weights"].unsqueeze(1).unsqueeze(-1).unsqueeze(-1).tile(1, multiplier, 1, 1, 1, 1) # Q x M x LQ x KQ x 1 x 1
                context_expert_weights = context_repr["expert_weights"].view(-1, multiplier, 1, 1, context_repr["expert_weights"].shape[1], context_repr["expert_weights"].shape[2]) # Q x M x 1 x 1 x LD x KD
                exact_match_weights = (query_expert_weights * context_expert_weights).permute((0, 2, 3, 1, 4, 5))  # Q * LQ * KQ * M * LD * KD
            else:  # inbatch
                exact_match_weights = query_repr["expert_weights"].unsqueeze(3).unsqueeze(4).unsqueeze(5) * context_repr["expert_weights"].unsqueeze(0).unsqueeze(1).unsqueeze(2)  # Q * LQ * KQ * D * LD * KD
            exact_match = torch.where(exact_match, exact_match_weights, torch.tensor([0.], dtype=exact_match_weights.dtype, device=exact_match_weights.device))
        else:
            exact_match = exact_match.to(query_repr["expert_repr"].dtype)
        scores = scores.unsqueeze(-1).unsqueeze(2)
        scores = scores * exact_match
        scores = scores.view(scores.shape[0], scores.shape[1] * scores.shape[2], scores.shape[3], scores.shape[4] * scores.shape[5])
        return scores

    def expert_sim_score(self, query_repr, context_repr, mask=None, pairwise=False):
        # For ColBERT, COIL, and CITADEL
        scores = self.colbert_score(query_repr, context_repr, pairwise)
        if "expert_ids" in query_repr:  # COIL and CITADEL
            if len(query_repr["expert_ids"].size()) == 2:  # COIL 
                scores = self.coil_score(scores, query_repr, context_repr, pairwise)
            else:  # CITADEL, exact_match -> routing
                scores = self.citadel_score(scores, query_repr, context_repr, pairwise)
        if self.query_pool == "sum":
            scores = scores.max(-1).values.sum(1) # Q x D or Q x M
        elif self.query_pool == "max":
            scores = scores.max(-1).values.max(1).values # Q x D or Q x M
        else:
            raise NotImplementedError("Invalid query pooling! Available: [max, sum]")

        if mask is not None:
            if pairwise:
                multiplier = context_repr["expert_repr"].shape[0] // query_repr["expert_repr"].shape[0] # number of hard negatives
                mask = mask.view(-1, multiplier) # Q x M
                scores[mask] = float("-inf")
            else:
                # mask is size num_ctx
                scores[mask.repeat(scores.size(0), 1)] = float("-inf")
        return scores
    
    def distilled_loss(self, input_logits, target_logits):
        input_logits = input_logits - input_logits.max(-1, True).values.detach() # numerical stability
        target_logits = target_logits - target_logits.max(-1, True).values.detach() # numerical stability
        
        input_probs = torch.softmax(input_logits, dim=-1)
        target_probs = torch.softmax(target_logits, dim=-1)
        loss = -(target_probs * torch.log(input_probs + 1e-6)).sum(-1).mean(0)
        return loss

    def router_loss(self, query_repr, context_repr, mask, pos_ctx_indices, teacher_scores):
        router_loss = 0.
        if 1 - self.teacher_coef > 0:
            router_scores = self.sim_score(query_repr["router_repr"], context_repr["router_repr"], mask, pairwise=not self.in_batch)
            if not self.in_batch:
                pos_ctx_indices = torch.zeros(len(router_scores), dtype=torch.int64).to(router_scores.device)
            router_loss = self.loss(router_scores, pos_ctx_indices)
        if self.teacher_coef > 0:
            pairwise_router_scores = self.sim_score(query_repr["router_repr"], 
                                                                context_repr["router_repr"], 
                                                                mask, pairwise=True)
            router_loss = (1 - self.teacher_coef) * router_loss + self.teacher_coef * self.distilled_loss(pairwise_router_scores/self.tau, teacher_scores/self.tau)
        self.log("train_router_loss", router_loss, prog_bar=True)
        return router_loss

    def expert_loss(self, query_repr, context_repr, mask, pos_ctx_indices, teacher_scores):
        expert_loss = 0.
        if 1 - self.teacher_coef > 0:
            expert_scores = 0.
            if "cls_repr" in context_repr:
                expert_scores += self.sim_score(query_repr["cls_repr"], context_repr["cls_repr"], mask)
            scores = self.expert_sim_score(query_repr, context_repr, mask, pairwise=not self.in_batch)
            expert_scores += scores
            if not self.in_batch:
                pos_ctx_indices = torch.zeros(len(expert_scores), dtype=torch.int64).to(expert_scores.device)
            expert_loss = self.loss(expert_scores, pos_ctx_indices)
        if self.teacher_coef > 0:
            pairwise_expert_scores = self.expert_sim_score(query_repr, 
                                                            context_repr, 
                                                            mask, pairwise=True)
            expert_loss = (1 - self.teacher_coef) * expert_loss + self.teacher_coef * self.distilled_loss(pairwise_expert_scores/self.tau, teacher_scores/self.tau)
        self.log("train_expert_loss", expert_loss, prog_bar=True)
        return expert_loss

    def compute_loss(self, query_repr, context_repr, mask, pos_ctx_indices, teacher_scores):
        loss = 0.
        if "router_repr" in context_repr: # training router contrastively
            router_loss = self.router_loss(query_repr, context_repr, mask, pos_ctx_indices, teacher_scores)
            loss += router_loss

        if "expert_repr" in context_repr: # training expert contrastively
            expert_loss = self.expert_loss(query_repr, context_repr, mask, pos_ctx_indices, teacher_scores)
            loss += expert_loss

        # regularization
        if self.query_router_marg_load_loss_coef > 0: # load balancing
            router_mask = query_repr["router_mask"]
            router_softmax_repr = query_repr["router_softmax_repr"]
            aux_loss = self.query_router_marg_load_loss_coef * (router_mask.mean(0) * router_softmax_repr.mean(0)).sum()
            aux_loss = self.anneal_func(aux_loss) if self.anneal_factor else aux_loss # ramping up
            self.log("train_query_router_marg_load_loss", aux_loss, prog_bar=True)
            loss += aux_loss
        if self.context_router_marg_load_loss_coef > 0: # load balancing
            router_mask = context_repr["router_mask"]
            router_softmax_repr = context_repr["router_softmax_repr"]
            aux_loss = self.context_router_marg_load_loss_coef * (router_mask.mean(0) * router_softmax_repr.mean(0)).sum()
            aux_loss = self.anneal_func(aux_loss) if self.anneal_factor else aux_loss # ramping up
            self.log("train_context_router_marg_load_loss", aux_loss, prog_bar=True)
            loss += aux_loss
        if self.context_expert_load_loss_coef > 0: # L1
            aux_loss = self.context_expert_load_loss_coef * context_repr["expert_weights"].sum(1).sum(1).mean(0)
            aux_loss = self.anneal_func(aux_loss) if self.anneal_factor else aux_loss
            self.log("train_context_expert_load_loss", aux_loss, prog_bar=True)
            loss += aux_loss
        if self.query_expert_load_loss_coef > 0: # L1
            aux_loss = self.query_expert_load_loss_coef * query_repr["expert_weights"].sum(1).sum(1).mean(0)
            aux_loss = self.anneal_func(aux_loss) if self.anneal_factor else aux_loss
            self.log("train_query_expert_load_loss", aux_loss, prog_bar=True)
            loss += aux_loss

        # Log other metrics, see details in models/CITADELEncoder.py
        if "avg_cond_num_experts" in context_repr:
            self.log("train_avg_context_cond_num_experts", context_repr["avg_cond_num_experts"].mean(), prog_bar=True)
        if "avg_marg_num_experts" in context_repr:
            self.log("train_avg_context_marg_num_experts", context_repr["avg_marg_num_experts"].mean(), prog_bar=True)
        if "avg_cond_num_experts" in query_repr:
            self.log("train_avg_query_cond_num_experts", query_repr["avg_cond_num_experts"].mean(), prog_bar=True)
        if "avg_marg_num_experts" in query_repr:
            self.log("train_avg_query_marg_num_experts", query_repr["avg_marg_num_experts"].mean(), prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        """
        This receives queries, each with mutliple contexts.
        """
        query_ids = batch["query_ids"]  # bs x tokens
        contexts_ids = batch["contexts_ids"]  # ctx_cnt x ctx_len
        pos_ctx_indices = batch["pos_ctx_indices"]  # bs
        mask = batch["ctx_mask"]  # ctx_cnt
        teacher_scores = batch["scores"]
        query_repr, context_repr = self(query_ids, contexts_ids)  # bs

        if self.cross_batch and isinstance(self.trainer.strategy, (DDPStrategy, DDPShardedStrategy)):
            query_repr, context_repr, mask, pos_ctx_indices, teacher_scores = self.distributed_gather(query_repr, context_repr, mask, pos_ctx_indices, teacher_scores)
        loss = self.compute_loss(query_repr, context_repr, mask, pos_ctx_indices, teacher_scores)
        return loss

    def _eval_step(self, batch, batch_idx):
        query_ids = batch["query_ids"]  # bs x tokens
        contexts_ids = batch["contexts_ids"]  # bs x ctx_cnt x ctx_len
        pos_ctx_indices = batch["pos_ctx_indices"]  # bs x ctx_cnt
        mask = batch["ctx_mask"]  # ctx_cnt
        query_repr, contexts_repr = self(query_ids, contexts_ids)
        pred_context_scores = self.expert_sim_score(query_repr, contexts_repr, mask)
        if "cls_repr" in query_repr:
            pred_context_scores += self.sim_score(query_repr["cls_repr"], contexts_repr["cls_repr"], mask)
        loss = self.loss(pred_context_scores, pos_ctx_indices)

        return (
            self.compute_rank_metrics(pred_context_scores, pos_ctx_indices),
            query_repr,
            contexts_repr,
            pos_ctx_indices,
            mask,
            loss,
        )

    def _eval_epoch_end(self, outputs, log_prefix="valid"):
        self.epoch += 1
        total_avg_rank, total_ctx_count, total_count = 0, 0, 0
        total_mrr = 0
        total_loss = 0
        total_score = 0
        
        for metrics, query_repr, contexts_repr, _, mask, loss in outputs:
            rank, mrr, score = metrics
            total_avg_rank += rank
            total_mrr += mrr
            total_score += score
            total_ctx_count += contexts_repr["expert_repr"].size(0) - torch.sum(mask)
            total_count += query_repr["expert_repr"].size(0)
            total_loss += loss
        total_ctx_count = total_ctx_count / len(outputs)
        total_loss = total_loss / len(outputs)
        
        metrics = {
            log_prefix + "_avg_rank": total_avg_rank / total_count,
            log_prefix + "_mrr": total_mrr / total_count,
            log_prefix + f"_accuracy@{self.k}": total_score / total_count,
            log_prefix + "_ctx_count": total_ctx_count,
            log_prefix + "_expert_loss": total_loss,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)
