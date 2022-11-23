# CITADEL:  Conditional Token Interaction via Dynamic Lexical Routing for Efficient and Effective Multi-Vector Retrieval

This page describes how to implement [CITADEL](https://arxiv.org/abs/2211.10411) with dpr-scale.
```
@article{li2022citadel,
title = {CITADEL: Conditional Token Interaction via Dynamic Lexical Routing for Efficient and Effective Multi-Vector Retrieval},
author = {Li, Minghan and Lin, Sheng-Chieh and Oguz, Barlas and Ghoshal, Asish and Lin, Jimmy and Mehdad, Yashar and Yih, Wen-tau and Chen, Xilun},
doi = {10.48550/ARXIV.2211.10411},
publisher = {arXiv},
year = {2022}
url = {https://arxiv.org/abs/2211.10411},
}
```
In the following, we describe how to train, encode, rerank, and retrieve with CITADEL on MS MARCO passage-v1 and TREC DeepLearning 2019/2020.
## Dependencies
First, make sure you have [Anaconda3](https://docs.anaconda.com/anaconda/install/index.html) installed.
Then use conda to create a new environment and activate it:
```
conda create -n dpr-scale python=3.8
conda activate dpr-scale
```
Now let's install the packages. First, follow the instructions [here](https://pytorch.org/get-started/locally/) to install PyTorch on your machine.
Then install faiss:
```
conda install -c conda-forge faiss-gpu
```
Finally install the packages in `requirement.txt`. Remember to comment out the packages in the .txt file that you've already installed to avoid conflicts.
```
pip install -r requirement.txt
```

## MS MARCO Passage-v1
### Data Prep
First, download the data from the [MS MARCO](https://microsoft.github.io/msmarco/) official website. Make sure to download and decompress the Collection, Qrels Train, Qrels Dev, and Queries.

Then, download and decompress the training data `train.jsonl.gz` from [Tevatron](https://huggingface.co/datasets/Tevatron/msmarco-passage/tree/main). We then split the training data into train and dev:
```
PYTHONPATH=. python dpr_scale/utils/prep_msmarco_exp.py --doc_path <train file path> --output_dir_path <output dir path>
```
By default we use 1\% training data as the validation set.

### Pre-trained Model Checkpoints

#### Checkpoints
- [CITADEL](https://dl.fbaipublicfiles.com/citadel/checkpoints/citadel/citadel/checkpoint_best.ckpt)

- [CITADEL+](https://dl.fbaipublicfiles.com/citadel/checkpoints/citadel/citadel_plus/checkpoint_best.ckpt)

#### Embeddings and Retrieval Index
- [CITADEL MS MARCO Index](https://dl.fbaipublicfiles.com/citadel/index/citadel_msmarco_index.tar.xz)

- [CITADEL+ MS MARCO Index](https://dl.fbaipublicfiles.com/citadel/index/citadel_plus_msmarco_index.tar.xz)


### Training
To train the model, run:

```
PYTHONPATH=.:$PYTHONPATH python dpr_scale/main.py -m \
--config-name msmarco_aws.yaml \
task=multivec task/model=citadel_model \
task.in_batch_eval=True datamodule.num_test_negative=10 trainer.max_epochs=10 \
task.model.tok_projection_dim=32 task.model.cls_projection_dim=128 \
task.shared_model=True +task.cross_batch=False +task.in_batch=True \
+task.add_cls=True \
+task.query_topk=1 +task.context_topk=5 \
+task.teacher_coef=0 +task.tau=1 \
+task.anneal_factor=0 \
+task.query_router_marg_load_loss_coef=1e-2 +task.context_router_marg_load_loss_coef=1e-2 \
+task.query_expert_load_loss_coef=0 +task.context_expert_load_loss_coef=1e-4 \
datamodule.batch_size=8 datamodule.num_negative=7 trainer.num_nodes=4 trainer.gpus=8
```

The default number of routing keys are `query_topk=1` and `context_topk=5` for query and passage, respectively. `router_marg_load_loss_coef` is to control the load balancing loss, while the `query_expert_load_loss_coef` is to control the L1 regularization. `anneal_factor` is to control the ramping up speed of the regularization coef, which we set to 0 in this case.

### Reranking
To quickly examine the quality of our trained model without the hassle of indexing, we could use the model to rerank the retrieved top-1000 candidates of BM25 and evaluate the results:
```
PATH_TO_OUTPUT_DIR=your_path_to_output_dir
CHECKPOINT_PATH=your_path_to_ckpt
DATA_PATH=/data_path/msmarco_passage/msmarco_corpus.tsv
PATH_TO_QUERIES_TSV=/data_path/msmarco_passage/dev_small.tsv
PATH_TO_TREC_TSV=/data_path/msmarco_passage/bm25.trec

PYTHONPATH=.:$PYTHONPATH python dpr_scale/citadel_scripts/run_reranking.py -m \
--config-name msmarco_aws.yaml \
task=multivec_rerank task/model=citadel_model \
task.model.tok_projection_dim=32 task.model.cls_projection_dim=128 \
task.shared_model=True \
+task.add_cls=True \
+task.query_topk=1 +task.context_topk=5 \
+task.output_dir=$PATH_TO_OUTPUT_DIR \
+task.checkpoint_path=$CHECKPOINT_PATH \
datamodule=generate_query_emb \
datamodule.test_path=$PATH_TO_TREC_TSV \
+datamodule.test_question_path=$PATH_TO_QUERIES_TSV \
+datamodule.query_trec=True \
+datamodule.test_passage_path=$DATA_PATH \
+topk=100 +cross_encoder=False \
+qrel_path=None \
+create_train_dataset=False \
+dataset=msmarco_passage
```
To get the `bm25.trec` file, please see the details [here](https://github.com/castorini/pyserini/blob/master/docs/experiments-msmarco-passage.md).

### Generate embeddings
If you are dealing with large corpus with million of documents, shard the corpus first before encoding.
Run the command with different shards in parallel:
```
CHECKPOINT_PATH=your_path_to_ckpt
for i in {0..5}
do 
    CTX_EMBEDDINGS_DIR=your_path_to_shard00${i}_embeddings
    DATA_PATH=/data_path/msmarco_passage/msmarco_corpus.00${i}.tsv
    PYTHONPATH=.:$PYTHONPATH python dpr_scale/citadel_scripts/generate_multivec_embeddings.py -m \
    --config-name msmarco_aws.yaml \
    datamodule=generate \
    task=multivec task/model=citadel_model \
    task.model.tok_projection_dim=32 task.model.cls_projection_dim=128 \
    +task.add_cls=True \
    +task.query_topk=1 +task.context_topk=1 \
    datamodule.test_path=$DATA_PATH \
    +task.ctx_embeddings_dir=$CTX_EMBEDDINGS_DIR \
    +task.checkpoint_path=$CHECKPOINT_PATH \
    +task.add_context_id=False > nohup${i}.log 2>&1&
done
```
The last argument `add_context_id` is for analysis if set `True`. 

### Merge embeddings
If using multiple gpus to encode the corpus, then we need to merge the embeddings and organize them into inverted index:
```
for i in {0..30000..10000} # indices range
do
    PYTHONPATH=.:$PYTHONPATH nohup python dpr_scale/citadel_scripts/merge_experts.py \
    your_path_to_shard00*_embeddings \
    $MERGED_CTX_EMBEDDINGS_DIR \
    "${i}-$(($i+10000))" > nohup_${i}.log 2>&1&
done
```

### Prune embeddings
To reduce the index size, we only keep the embeddings with routing weights larger than some threshold:
```
pruning_weight=0.9 # default
PYTHONPATH=.:$PYTHONPATH python dpr_scale/citadel_scripts/prune_experts.py \
$MERGED_CTX_EMBEDDINGS_DIR/expert \
$MERGED_CTX_EMBEDDINGS_DIR \
$pruning_weight \ 
"0-31000" # index range
```
The output is at `${MERGED_CTX_EMBEDDINGS_DIR}/expert_pruned${pruning_weight}`

### Quantize embeddings
If you want to further compress the embeddings, use our custom product quantization module:
```
PYTHONPATH=.:$PYTHONPATH python dpr_scale/citadel_scripts/run_quantization.py -m \
--config-name msmarco_aws.yaml \
task=multivec task/model=citadel_model \
+task.ctx_embeddings_dir=$MERGED_CTX_EMBEDDINGS_DIR/expert_pruned${weight} \
+task.output_dir=$MERGED_CTX_EMBEDDINGS_DIR/expert_pruned${weight}_pq_nbits2 \
+cls_dim=128 +dim=32 \
+sub_vec_dim=4 +num_centroids=256 +iter=5 \ 
+cuda=True \ 
+threshold=$threshold \ 
trainer=gpu_1_host trainer.gpus=1
```
You could change the `sub_vec_dim` from 4 to 8 to get nbits=1 compression. We don't recommend changing other parameters unless you know what you are doing.

### Retrieval
Build Cython extension for fast retrieval on CPU. Please see [here](https://github.com/luyug/COIL/tree/main/retriever#fast-retriver) for details.
```
cd dpr-scale/retriever_ext
pip install Cython
python setup.py build_ext --inplace
```

To run retrieval on the compressed corpus embeddings, use:
```
PORTION=1.0 # move how much portion of indexes to gpu
DATA_PATH=/data_path/msmarco_passage/msmarco_corpus.tsv
PATH_TO_QUERIES_TSV=/data_path/msmarco_passage/dev_small.tsv
CTX_EMBEDDINGS_DIR=$MERGED_CTX_EMBEDDINGS_DIR/expert_pruned${weight}_pq_nbits2
CHECKPOINT_PATH=your_path_to_ckpt
OUTPUT_DIR=your_path_to_output_dir

HYDRA_FULL_ERROR=1 PYTHONPATH=.:$PYTHONPATH python dpr_scale/citadel_scripts/run_citadel_retrieval.py \
--config-name msmarco_aws.yaml \
datamodule=generate_multivec_query_emb \
+datamodule.trec_format=True \
datamodule.test_path=$PATH_TO_QUERIES_TSV \
datamodule.test_batch_size=1 \
task=multivec_retrieval task/model=citadel_model \
task.model.tok_projection_dim=32 \
task.model.cls_projection_dim=128 +task.add_cls=True task.shared_model=True \
+task.query_topk=1 +task.context_topk=$k \
+task.output_path=$OUTPUT_DIR \
+task.ctx_embeddings_dir=$CTX_EMBEDDINGS_DIR \
+task.checkpoint_path=$CHECKPOINT_PATH \
+task.passages=$DATA_PATH \
+task.portion=$PORTION \
+task.topk=1000 \
+task.cuda=True \
+task.quantizer=pq \
+task.sub_vec_dim=4 \
trainer.precision=16 \
+task.expert_parallel=True \
trainer=gpu_1_host trainer.gpus=1
```
The output is a trec file in the output dir. Query embeddings are automatically saved in the output_dir.

To run without quantization, use:
```
PORTION=1.0 # move how much portion of indexes to gpu
DATA_PATH=/data_path/msmarco_passage/msmarco_corpus.tsv
PATH_TO_QUERIES_TSV=/data_path/msmarco_passage/dev_small.tsv
CTX_EMBEDDINGS_DIR=$MERGED_CTX_EMBEDDINGS_DIR/expert_pruned${weight}
CHECKPOINT_PATH=your_path_to_ckpt
OUTPUT_DIR=your_path_to_output_dir

HYDRA_FULL_ERROR=1 PYTHONPATH=.:$PYTHONPATH python dpr_scale/citadel_scripts/run_citadel_retrieval.py \
--config-name msmarco_aws.yaml \
datamodule=generate_multivec_query_emb \
+datamodule.trec_format=True \
datamodule.test_path=$PATH_TO_QUERIES_TSV \
datamodule.test_batch_size=1 \
task=multivec_retrieval task/model=citadel_model \
task.model.tok_projection_dim=32 \
task.model.cls_projection_dim=128 +task.add_cls=True task.shared_model=True \
+task.query_topk=1 +task.context_topk=$k \
+task.output_path=$OUTPUT_DIR \
+task.ctx_embeddings_dir=$CTX_EMBEDDINGS_DIR \
+task.checkpoint_path=$CHECKPOINT_PATH \
+task.passages=$DATA_PATH \
+task.portion=$PORTION \
+task.topk=1000 +task.cuda=True \
+task.quantizer=None \
+task.sub_vec_dim=4 \
trainer.precision=16 \
+task.expert_parallel=True \
trainer=gpu_1_host trainer.gpus=1
```

You could change `trainer.gpus` to run retrieval on multiple gpus.
Set `cuda=False` if you want to use cpu for retrieval. Further set `trainer.gpus=0` to use cpu for query encoding as well.

### Get evaluation metrics for MSMARCO
This python script uses pytrec_eval in background:
```
python dpr_scale/citadel_scripts/msmarco_eval.py /data_path/data/msmarco_passage/qrels.dev.small.tsv PATH_TO_OUTPUT_TREC_FILE
```

### Get evaluation metrics for TREC DeepLearning 2019 and 2020
We use [pyserini](https://github.com/castorini/pyserini) to evaluate on trec dl. Feel free to use pytrec_eval as well. The reason is that we need to deal with qrels with different relevance levels in TREC DL. If you plan to use pyserini, please install it in a different environment to avoid package conflicts with dpr-scale.
```
# Recall
python -m pyserini.eval.trec_eval -c -mrecall.1000 -l 2 /data_path/trec_dl/2019qrels-pass.txt PATH_TO_OUTPUT_TREC_FILE

# nDCG@10
python -m pyserini.eval.trec_eval -c -mndcg_cut.10 /data_path/trec_dl/2019qrels-pass.txt PATH_TO_OUTPUT_TREC_FILE
```

## BEIR
We will use the ckpt trained on MS MARCO passsage-v1 to evaulate on 13 datasets in [BEIR](https://github.com/beir-cellar/beir).

### Data Prep
First we need to download the beir datasets and convert the data into dpr format:
```
DATASET=(arguana climate-fever dbpedia-entity fever fiqa hotpotqa nfcorpus nq quora scifact scidocs trec-covid webis-touche2020)
for dataset in ${DATASET[*]}
do
echo $dataset
python dpr_scale/convert_beir_to_dpr_format.py $dataset <output path>
done
```

### Generate embeddings
We then encode the corpus of each dataset. For datasets with large corpus, we split it into multiple shards:
```
CHECKPOINT_PATH=your_path_to_ckpt
DATASET=(arguana nfcorpus fiqa quora scidocs scifact trec-covid webis-touche2020)
for dataset in ${DATASET[*]} 
do
    echo $dataset
    CTX_EMBEDDINGS_DIR=your_path_to_output_embedding_dir
    DATA_PATH=/data_path/beir/datasets/${dataset}/dpr-scale/corpus.tsv

    HYDRA_FULL_ERROR=1 PYTHONPATH=.:$PYTHONPATH nohup python dpr_scale/citadel_scripts/generate_multivec_embeddings.py -m --config-name msmarco_aws.yaml \
    datamodule=generate \
    datamodule.test_path=$DATA_PATH \
    task=multivec task/model=citadel_model \
    task.model.tok_projection_dim=32 task.model.cls_projection_dim=128 \
    task.shared_model=True \
    +task.add_cls=True \
    +task.query_topk=1 +task.context_topk=5 \
    +task.weight_threshold=0.0 \
    +task.ctx_embeddings_dir=$CTX_EMBEDDINGS_DIR \
    +task.checkpoint_path=$CHECKPOINT_PATH \
    +task.add_context_id=False > nohup_${dataset}.log 2>&1&
done

DATASET=(climate-fever dbpedia-entity fever hotpotqa nq)
SHARD=(0 1 2)
for dataset in ${DATASET[*]} 
do
    echo $dataset
    for shard in ${SHARD[*]}
    do
    CTX_EMBEDDINGS_DIR=your_path_to_output_shard_embeddings
    DATA_PATH=/data_path/beir/datasets/${dataset}/dpr-scale/corpus.00$shard.tsv

    HYDRA_FULL_ERROR=1 PYTHONPATH=.:$PYTHONPATH nohup python dpr_scale/citadel_scripts/generate_multivec_embeddings.py -m --config-name msmarco_aws.yaml \
    datamodule=generate \
    datamodule.test_path=$DATA_PATH \
    task=multivec task/model=citadel_model \
    task.model.tok_projection_dim=32 task.model.cls_projection_dim=128 \
    task.shared_model=True +task.cross_batch=False +task.in_batch=True \
    +task.add_cls=True \
    +task.query_topk=1 +task.context_topk=5 \
    +task.weight_threshold=0.0 \
    +task.ctx_embeddings_dir=$CTX_EMBEDDINGS_DIR \
    +task.checkpoint_path=$CHECKPOINT_PATH \
    +task.add_context_id=False > nohup_${dataset}_${shard}.log 2>&1&
    done
done
```

### Merge embeddings
```
DATASET=(arguana nfcorpus fiqa quora scidocs scifact trec-covid webis-touche2020 climate-fever dbpedia-entity fever hotpotqa nq)
for dataset in ${DATASET[*]} 
do
echo $dataset
OUTPUT_DIR=your_path_to_merged_embeddings
CTX_EMBEDDINGS_DIR=your_path_to_embedding_dirs
PYTHONPATH=.:$PYTHONPATH python dpr_scale/citadel_scripts/merge_experts.py $OUTPUT_DIR "$CTX_EMBEDDINGS_DIR" "0-31000"
done
```
You could further compress the index size using product quantization and pruning. We skip the compression step for simplicity.

### Retrieval
```
dataset=$1
OUTPUT_DIR=path_to_retrieval_output_dir
CTX_EMBEDDINGS_DIR=path_to_merged_embeddings/expert
CHECKPOINT_PATH=path_to_your_ckpt

I2D_PATH=/data_path/beir/datasets/${dataset}/dpr-scale/index2docid.tsv
DATA_PATH=/data_path/beir/datasets/${dataset}/dpr-scale/corpus.tsv
PATH_TO_QUERIES_TSV=/data_path/beir/datasets/${dataset}/dpr-scale/queries.tsv

PORTION=0.001 # how much portion of the index should be moved to GPU before retrieval

HYDRA_FULL_ERROR=1 PYTHONPATH=.:$PYTHONPATH python dpr_scale/citadel_scripts/run_citadel_retrieval.py \
--config-name msmarco_aws.yaml \
datamodule=generate_multivec_query_emb \
datamodule.test_path=$PATH_TO_QUERIES_TSV \
datamodule.test_batch_size=1 \
+datamodule.trec_format=True \
task=multivec_retrieval task/model=citadel_model \
task.model.tok_projection_dim=32 \
task.model.cls_projection_dim=128 +task.add_cls=True task.shared_model=True \
+task.query_topk=1 +task.context_topk=5 \
+task.output_path=$OUTPUT_DIR \
+task.ctx_embeddings_dir=$CTX_EMBEDDINGS_DIR \
+task.checkpoint_path=$CHECKPOINT_PATH \
+task.index2docid_path=$I2D_PATH \
+task.passages=$DATA_PATH \
+task.portion="$PORTION" \
+task.topk=1000 +task.cuda=True +task.quantizer=None +task.sub_vec_dim=4 trainer.precision=16 +task.expert_parallel=True \
trainer=gpu_1_host trainer.gpus=1
```

### Evaluation
You could run evaluation on beir retrieval results using:
```
DATASET=(arguana climate-fever dbpedia-entity fever fiqa hotpotqa nfcorpus nq quora scifact scidocs trec-covid webis-touche2020)
for dataset in ${DATASET[*]} 
do
echo $dataset
QRELS_PATH=/data_path/beir/datasets/${dataset}/dpr-scale/test.tsv
TREC_PATH=path_to_retrieval_output_dir/retrieval.trec
python dpr_scale/citadel_scripts/run_beir_eval.py $QRELS_PATH $TREC_PATH > /data_path/results/beir/${dataset}/eval_results.txt
done
```

## License
The majority of CITADEL is licensed under CC-BY-NC, however portions of the project are available under separate license terms: code from the [COIL](https://github.com/luyug/COIL) project is licensed under the Apache 2.0 license.
