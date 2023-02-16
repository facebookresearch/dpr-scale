Scalable implementation of dense retrieval.

## This repo implements the following papers:

- [Domain-matched Pre-training Tasks for Dense Retrieval](https://arxiv.org/abs/2107.13602)
- [Salient Phrase Aware Dense Retrieval: Can a Dense Retriever Imitate a Sparse One?](https://arxiv.org/abs/2110.06918)
    - https://github.com/facebookresearch/dpr-scale/tree/main/spar
- [CITADEL: Conditional Token Interaction via Dynamic Lexical Routing for Efficient and Effective Multi-Vector Retrieval](https://arxiv.org/abs/2211.10411)
    - https://github.com/facebookresearch/dpr-scale/tree/citadel
- [How to Train Your DRAGON: Diverse Augmentation Towards Generalizable Dense Retrieval](https://arxiv.org/abs/2302.07452)
    - https://github.com/facebookresearch/dpr-scale/tree/main/dragon


## Input Data Format (JSONL)
Linewise JSON file where each row typically looks like:
```
{
    "question": ...,
    "positive_ctxs": [
        {
        "title": "...",
        "text": "....",
        <optional>
        "id": ...,
        "relevance": ...
        }, {...}, ...
    ],
    "hard_negative_ctxs": [{...}, {...}, ...]
}
```

or
```
{
    "question": ...,
    "id": ...,
    "ctxs": [
        {
        "has_answer": True or False,    
        "title": "...",
        "text": "....",
        <optional>
        "id": ...,
        "relevance": ...
        }, {...}, ...
    ]
}
```

If your training data is large, you can use a lightweight format by specifying the line number (`docidx` starting from 0) of the document in the corpus without storing its title and text:
```
{
    "question": ...,
    "positive_ctxs": [
        {
        "docidx": ..., # denote the position of the passage in the corpus, starting from 0
        <optional>
        "id": ...,
        "relevance": ...
        }, {...}, ...
    ],
    "hard_negative_ctxs": [{...}, {...}, ...]
}
```
This format requires you to use `DenseRetrieverMultiJsonlDataModule` and set `--corpus_path` while training. See below example with config `msmarco_baseline.yaml`. The corpus format follow the default Wiki corpus format with header at first line:
```
id"\t"text"\t"title
<id>"\t"<text>"\t"<title>
...
```

## Training on cluster

By default it trains locally:

```
PYTHONPATH=.:$PYTHONPATH python dpr_scale/main.py trainer.gpus=1
``` 

You can try our example of baseline training locally on MS MARCO dataset with the lightweight data format:
```
PYTHONPATH=.:$PYTHONPATH python dpr_scale/main.py -m --config-name msmarco_baseline.yaml 
```

### SLURM Training

To train the model on SLURM, run:

```
PYTHONPATH=.:$PYTHONPATH python dpr_scale/main.py -m trainer=slurm trainer.num_nodes=2 trainer.gpus=2
```

### Reproduce DPR on 8 gpus
```
PYTHONPATH=.:$PYTHONPATH python dpr_scale/main.py -m --config-name nq.yaml  +hydra.launcher.name=dpr_stl_nq_reproduce
```

### Generate embeddings on Wikipedia
```
PYTHONPATH=.:$PYTHONPATH python dpr_scale/generate_embeddings.py -m --config-name nq.yaml datamodule=generate datamodule.test_path=psgs_w100.tsv +task.ctx_embeddings_dir=<CTX_EMBEDDINGS_DIR> +task.checkpoint_path=<CHECKPOINT_PATH>
```

### Get retrieval results
Currently this runs on 1 GPU.  Use CTX_EMBEDDINGS_DIR from above.
```
PYTHONPATH=.:$PYTHONPATH python dpr_scale/run_retrieval.py --config-name nq.yaml trainer=gpu_1_host trainer.gpus=1 +task.output_path=<PATH_TO_OUTPUT_JSON> +task.ctx_embeddings_dir=<CTX_EMBEDDINGS_DIR> +task.checkpoint_path=<CHECKPOINT_PATH> +task.passages=psgs_w100.tsv datamodule.test_path=<PATH_TO_QUERIES_JSONL>
```

### Generate query embeddings
Alternatively, query embedding generation and retrieval can be separated.
After query embeddings are generated using the following command, the `run_retrieval_fb.py` or `run_retrieval_multiset.py` script can be used to perform retrieval.
```
PYTHONPATH=.:$PYTHONPATH python dpr_scale/generate_query_embeddings.py -m --config-name nq.yaml trainer.gpus=1 datamodule.test_path=<PATH_TO_QUERIES_JSONL> +task.ctx_embeddings_dir=<CTX_EMBEDDINGS_DIR> +task.checkpoint_path=<CHECKPOINT_PATH> +task.query_emb_output_path=<OUTPUT_TO_QUERY_EMB>
```

### Get evaluation metrics for a given JSON output file
```
python dpr_scale/eval_dpr.py --retrieval <PATH_TO_OUTPUT_JSON> --topk 1 5 10 20 50 100 
```

### Get evaluation metrics for MSMARCO
python dpr_scale/msmarco_eval.py ~data/msmarco/qrels.dev.small.tsv PATH_TO_OUTPUT_JSON

# Domain-matched Pre-training Tasks for Dense Retrieval
`Paper: https://arxiv.org/abs/2107.13602`

The sections below provide links to datasets and pretrained models, as well as, instructions to prepare datasets, pretrain and fine-tune them.

## Q&A Datasets
### PAQ
Download the dataset from [here](https://dl.fbaipublicfiles.com/dpr_scale/paq/PAQ.dpr.train.neg1.jsonl.zip)

## Conversational Datasets
You can download the dataset from the respective tables.
### Reddit
File | Download Link
|---|---
train | [download](https://dl.fbaipublicfiles.com/dpr_scale/reddit/train.200M.jsonl)
dev | [download](https://dl.fbaipublicfiles.com/dpr_scale/reddit/dev.jsonl)


### ConvAI2
File | Download Link
|---|---
train | [download](https://dl.fbaipublicfiles.com/dpr_scale/convai2/train.jsonl)
dev | [download](https://dl.fbaipublicfiles.com/dpr_scale/convai2/valid.jsonl)


### DSTC7
File | Download Link
|---|---
train | [download](https://dl.fbaipublicfiles.com/dpr_scale/dstc7/ubuntu_train.jsonl)
dev | [download](https://dl.fbaipublicfiles.com/dpr_scale/dstc7/ubuntu_dev.jsonl)
test | [download](https://dl.fbaipublicfiles.com/dpr_scale/dstc7/ubuntu_test.jsonl)

Prepare by downloading the tar ball linked [here](https://github.com/facebookresearch/ParlAI/blob/master/parlai/tasks/dstc7/build.py), and using the command below.
```
DSTC7_DATA_ROOT=<path_of_dir_where_the_data_is_extracted>
python dpr_scale/data_prep/prep_conv_datasets.py \
    --dataset dstc7 \
    --in_file_path $DSTC7_DATA_ROOT/ubuntu_train_subtask_1_augmented.json \
    --out_file_path $DSTC7_DATA_ROOT/ubuntu_train.jsonl
```

### Ubuntu V2
File | Download Link
|---|---
train | [download](https://dl.fbaipublicfiles.com/dpr_scale/ubuntu/train.jsonl)
dev | [download](https://dl.fbaipublicfiles.com/dpr_scale/ubuntu/valid.jsonl)
test | [download](https://dl.fbaipublicfiles.com/dpr_scale/ubuntu/test.jsonl)

Prepare by downloading the tar ball linked [here](https://github.com/facebookresearch/ParlAI/blob/master/parlai/tasks/ubuntu/build.py), and using the command below.
```
UBUNTUV2_DATA_ROOT=<path_of_dir_where_the_data_is_extracted>
python dpr_scale/data_prep/prep_conv_datasets.py \
    --dataset ubuntu2 \
    --in_file_path $UBUNTUV2_DATA_ROOT/train.csv \
    --out_file_path $UBUNTUV2_DATA_ROOT/train.jsonl
```

## Pretraining DPR
### Pretrained Checkpoints
Pretrained Model | Dataset | Download Link
|---|---|---
BERT-base | PAQ | [download](https://dl.fbaipublicfiles.com/dpr_scale/checkpoints/paq_bert_base.ckpt)
BERT-large | PAQ | [download](https://dl.fbaipublicfiles.com/dpr_scale/checkpoints/paq_bert_large.ckpt)
BERT-base | Reddit | [download](https://dl.fbaipublicfiles.com/dpr_scale/checkpoints/reddit_bert_base.ckpt)
BERT-large | Reddit | [download](https://dl.fbaipublicfiles.com/dpr_scale/checkpoints/reddit_bert_large.ckpt)
RoBERTa-base | Reddit | [download](https://dl.fbaipublicfiles.com/dpr_scale/checkpoints/reddit_roberta_base.ckpt)
RoBERTa-large | Reddit | [download](https://dl.fbaipublicfiles.com/dpr_scale/checkpoints/reddit_roberta_large.ckpt)

### Pretraining on PAQ dataset
```
DPR_ROOT=<path_of_your_repo's_root>
MODEL="bert-large-uncased"
NODES=8
BSZ=16
MAX_EPOCHS=20
LR=1e-5
TIMOUT_MINS=4320
EXP_DIR=<path_of_the_experiment_dir>
TRAIN_PATH=<path_of_the_training_data_file>
mkdir -p ${EXP_DIR}/logs
PYTHONPATH=$DPR_ROOT python ${DPR_ROOT}/dpr_scale/main.py -m \
    --config-dir ${DPR_ROOT}/dpr_scale/conf \
    --config-name nq.yaml \
    hydra.launcher.timeout_min=$TIMOUT_MINS \
    hydra.sweep.dir=${EXP_DIR} \
    trainer.num_nodes=${NODES} \
    task.optim.lr=${LR} \
    task.model.model_path=${MODEL} \
    trainer.max_epochs=${MAX_EPOCHS} \
    datamodule.train_path=$TRAIN_PATH \
    datamodule.batch_size=${BSZ} \
    datamodule.num_negative=1 \
    datamodule.num_val_negative=10 \
    datamodule.num_test_negative=50 > ${EXP_DIR}/logs/log.out 2> ${EXP_DIR}/logs/log.err &
```

### Pretraining on Reddit dataset
```
# Use a batch size of 16 for BERT and RoBERTa base models.
BSZ=4
NODES=8
MAX_EPOCHS=5
WARMUP_STEPS=10000
LR=1e-5
MODEL="roberta-large"
EXP_DIR=<path_of_the_experiment_dir>
PYTHONPATH=. python dpr_scale/main.py -m \
    --config-dir ${DPR_ROOT}/dpr_scale/conf \
    --config-name reddit.yaml \
    hydra.launcher.nodes=${NODES} \
    hydra.sweep.dir=${EXP_DIR} \
    trainer.num_nodes=${NODES} \
    task.optim.lr=${LR} \
    task.model.model_path=${MODEL} \
    trainer.max_epochs=${MAX_EPOCHS} \
    task.warmup_steps=${WARMUP_STEPS} \
    datamodule.batch_size=${BSZ} > ${EXP_DIR}/logs/log.out 2> ${EXP_DIR}/logs/log.err &
```

## Fine-tuning DPR on downstream tasks/datasets
### Fine-tune the pretrained PAQ checkpoint
```
# You can also try 2e-5 or 5e-5. Usually these 3 learning rates work best.
LR=1e-5
# Use a batch size of 32 for BERT and RoBERTa base models.
BSZ=12
MODEL="bert-large-uncased"
MAX_EPOCHS=40
WARMUP_STEPS=1000
NODES=1
PRETRAINED_CKPT_PATH=<path_of_checkpoint_pretrained_on_reddit>
EXP_DIR=<path_of_the_experiment_dir>
PYTHONPATH=. python dpr_scale/main.py -m \
    --config-dir ${DPR_ROOT}/dpr_scale/conf \
    --config-name nq.yaml \
    hydra.launcher.name=${NAME} \
    hydra.sweep.dir=${EXP_DIR} \
    trainer.num_nodes=${NODES} \
    trainer.max_epochs=${MAX_EPOCHS} \
    datamodule.num_negative=1 \
    datamodule.num_val_negative=25 \
    datamodule.num_test_negative=50 \
    +trainer.val_check_interval=150 \
    task.warmup_steps=${WARMUP_STEPS} \
    task.optim.lr=${LR} \
    task.pretrained_checkpoint_path=$PRETRAINED_CKPT_PATH \
    task.model.model_path=${MODEL} \
    datamodule.batch_size=${BSZ} > ${EXP_DIR}/logs/log.out 2> ${EXP_DIR}/logs/log.err &
```

### Fine-tune the pretrained Reddit checkpoint
Batch sizes that worked on Volta 32GB GPUs for respective model and datasets.
Model | Dataset | Batch Size
|---|---|---
BERT/RoBERTa base | ConvAI2 | 64
RBERT/RoBERTa base | ConvAI2 | 16
BERT/RoBERTa base | DSTC7 | 24
BERT/RoBERTa base | DSTC7 | 8
BERT/RoBERTa base | Ubuntu V2 | 64
BERT/RoBERTa large | Ubuntu V2 | 16
```
# Change the config file name to convai2.yaml or dstc7.yaml for the respective datasets.
CONFIG_FILE_NAME=ubuntuv2.yaml
# You can also try 2e-5 or 5e-5. Usually these 3 learning rates work best.
LR=1e-5
BSZ=16
NODES=1
MAX_EPOCHS=5
WARMUP_STEPS=10000
MODEL="roberta-large"
PRETRAINED_CKPT_PATH=<path_of_checkpoint_pretrained_on_reddit>
EXP_DIR=<path_of_the_experiment_dir>
PYTHONPATH=${DPR_ROOT} python ${DPR_ROOT}/dpr_scale/main.py -m \
    --config-dir=${DPR_ROOT}/dpr_scale/conf \
    --config-name=$CONFIG_FILE_NAME \
    hydra.launcher.nodes=${NODES} \
    hydra.sweep.dir=${EXP_DIR} \
    trainer.num_nodes=${NODES} \
    trainer.max_epochs=${MAX_EPOCHS} \
    +trainer.val_check_interval=150 \
    task.pretrained_checkpoint_path=$PRETRAINED_CKPT_PATH \
    task.warmup_steps=${WARMUP_STEPS} \
    task.optim.lr=${LR} \
    task.model.model_path=$MODEL \
    datamodule.batch_size=${BSZ} > ${EXP_DIR}/logs/log.out 2> ${EXP_DIR}/logs/log.err &
```

## License
dpr-scale is CC-BY-NC 4.0 licensed as of now.
