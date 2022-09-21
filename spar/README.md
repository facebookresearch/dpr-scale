# SPAR: Salient Phrase Aware Dense Retriever

This repo contains source code and pre-trained models for our paper:

[**Salient Phrase Aware Dense Retrieval: Can a Dense Retriever Imitate a Sparse One?**](https://arxiv.org/abs/2110.06918)
<br>
Xilun Chen, Kushal Lakhotia, Barlas Oğuz, Anchit Gupta, Patrick Lewis, Stan Peshterliev, Yashar Mehdad, Sonal Gupta and Wen-tau Yih
<br>
**Meta AI**


## Pre-trained Checkpoints

**News**: The pre-trained checkpoints are now available on Huggingface! See [here](https://huggingface.co/facebook/spar-wiki-bm25-lexmodel-query-encoder) for how to get the models using the Huggingface `transformers` library.

Pretrained Model | Corpus | Teacher | Architecture | Download Link | Huggingface Path: Query Encoder | Huggingface Path: Context Encoder
|---|---|---|---|---|---|---
Wiki BM25 Λ | Wikipedia | BM25 | BERT-base | [download](https://dl.fbaipublicfiles.com/SPAR/checkpoints/wiki_lambda_bm25_bert_base.ckpt) | [facebook/spar-wiki-bm25-lexmodel-query-encoder](https://huggingface.co/facebook/spar-wiki-bm25-lexmodel-query-encoder) | [facebook/spar-wiki-bm25-lexmodel-context-encoder](https://huggingface.co/facebook/spar-wiki-bm25-lexmodel-context-encoder)
PAQ BM25 Λ | PAQ | BM25 | BERT-base | [download](https://dl.fbaipublicfiles.com/SPAR/checkpoints/paq_lambda_bm25_bert_base.ckpt) | [facebook/spar-wiki-bm25-lexmodel-query-encoder](https://huggingface.co/facebook/spar-paq-bm25-lexmodel-query-encoder) | [facebook/spar-wiki-bm25-lexmodel-context-encoder](https://huggingface.co/facebook/spar-paq-bm25-lexmodel-context-encoder)
MARCO BM25 Λ | MS MARCO | BM25 | BERT-base | [download](https://dl.fbaipublicfiles.com/SPAR/checkpoints/marco_lambda_bm25_bert_base.ckpt) | [facebook/spar-marco-bm25-lexmodel-query-encoder](https://huggingface.co/facebook/spar-marco-bm25-lexmodel-query-encoder) | [facebook/spar-marco-bm25-lexmodel-context-encoder](https://huggingface.co/facebook/spar-marco-bm25-lexmodel-context-encoder)
MARCO UniCOIL Λ | MS MARCO | UniCOIL | BERT-base | [download](https://dl.fbaipublicfiles.com/SPAR/checkpoints/marco_lambda_unicoil_bert_base.ckpt) | [facebook/spar-marco-unicoil-lexmodel-query-encoder](https://huggingface.co/facebook/spar-marco-unicoil-lexmodel-query-encoder) | [facebook/spar-marco-unicoil-lexmodel-context-encoder](https://huggingface.co/facebook/spar-marco-unicoil-lexmodel-context-encoder)

For convenience, we also provide a checkpoint to the [Multiset DPR model](https://dl.fbaipublicfiles.com/SPAR/checkpoints/dpr_multiset_bert_base.ckpt), which is the same model as [this one](https://dl.fbaipublicfiles.com/dpr/checkpoint/retriver/multiset/hf_bert_base.cp) but converted to the dpr-scale format.

## Example SPAR Workflow: Open-Domain Question Answering
1. Prepare necessary data including the document collection (Wikipedia passages) and evluation datasets (NQ, TriviaQA, etc.)
1. Pick your favorite retriever (e.g. DPR) as the base retriever in SPAR
1. Train a Λ model (or use one of the *pre-trained* ones above and skip the first three steps)
    1. Pick a sparse teacher (e.g. BM25)
    1. [Generate training data for Λ](#generate-lambda-training-data)
    1. [Train Λ using `dpr-scale`](#lambda-training)
1. [Generate passage and query embeddings for both Λ and the base retriever](#generate-embeddings)
1. [Tune the concontenation weight in SPAR on the dev set](#tune-the-spar-concatenation-weight)
1. [Perform SPAR retrieval using the concatenated embeddings](#run-spar-retrieval)

### Generate Λ Training Data <a name="generate-lambda-training-data"></a>

**It is recommended to use one of the pre-trained Λ checkpoints, as the generation of training data and the training itself can be rather expensive.**

To generate your own training data for Λ, first generate the training queries using the following script:
```bash
# make sure you're in the root folder of the dpr-scale repo

PYTHONPATH=. python dpr_scale/utils/prep_wiki_exp.py --workers 16 --doc_path <PATH_TO_DPR_PASSAGES>/psgs_w100.tsv
```

Then use the teacher retriever to generate the positive and hard negative passages for the training queries, and store the output in the dpr-scale JSONL format.
In the paper, we use Pyserini to retrieve the top 50 passages for each training query, and use the top 10 as the positive, and the bottom 5 as hard negatives.

### Λ Training <a name="lambda-training"></a>
The following script trains the Wikipedia BM25 Λ.

```bash
# make sure you're in the root folder of the dpr-scale repo

PYTHONPATH=.:$PYTHONPATH python dpr_scale/main.py -m \
    --config-name nq.yaml \
    +hydra.launcher.name=spar_wiki_bm25_lambda_training \
    trainer.num_nodes=8 \
    trainer.max_epochs=20 \
    trainer.log_every_n_steps=10 \
    +trainer.val_check_interval=500 \
    task.warmup_steps=10000 \
    task.optim.lr=3e-05 \
    task.model.model_path=bert-base-uncased \
    task.in_batch_eval=False \
    +datamodule.pos_ctx_sample=True \
    datamodule.num_negative=1 \
    datamodule.num_val_negative=1 \
    datamodule.num_test_negative=50 \
    datamodule.train_path=<PATH_TO_LAMBDA_TRAINING_DATA> \
    datamodule.val_path=<PATH_TO_NQ_DEV_WITH_BM25_AS_LABEL> \
    datamodule.test_path=<PATH_TO_NQ_DEV_WITH_BM25_AS_LABEL> \
    datamodule.batch_size=32
```

### Generate embeddings
For both Λ and the base retriever, the passage and query embeddings need to be generated and retrieval also needs to be performed to produce a JSON results file for each dataset, before we can tune the concatenation weight in SPAR.

Detailed instructions for generating embeddings and evaluating a given retriever are given in the parent folder.
Below, we provide an example script to generate embeddings and evaluate the pre-trained `Wiki BM25 Λ` model.

```bash
# make sure you're in the root folder of the dpr-scale repo
# First download the Wiki BM25 Λ checkpoint to ./spar/checkpoints/wiki_lambda_bm25_bert_base.ckpt

CHECKPOINT_PATH="./spar/checkpoints/wiki_lambda_bm25_bert_base.ckpt"
CTX_EMBEDDINGS_DIR="./spar/outputs/wiki_lambda_bm25_bert_base/"
# generate passage embeddings
PYTHONPATH=.:$PYTHONPATH python dpr_scale/generate_embeddings.py -m \
    --config-name nq.yaml \
    datamodule=generate \
    datamodule.test_path=<PATH_TO_DPR_PASSAGES>/psgs_w100.tsv \
    +task.ctx_embeddings_dir=$CTX_EMBEDDINGS_DIR \
    +task.checkpoint_path=$CHECKPOINT_PATH


# generate query embeddings
for dataset in nq_test squad1_test webq_test trivia_test trec_test nq_dev squad1_dev webq_dev trivia_dev trec_dev
do
        PYTHONPATH=.:$PYTHONPATH python dpr_scale/generate_query_embeddings.py -m \
            --config-name nq.yaml \
            trainer=slurm \
            trainer.gpus=1 \
            +task.ctx_embeddings_dir=$CTX_EMBEDDINGS_DIR \
            +task.checkpoint_path=$CHECKPOINT_PATH \
            +task.passages=<PATH_TO_DPR_PASSAGES>/psgs_w100.tsv \
            datamodule.test_path=<PATH_TO_DPR_DATA>/$dataset.jsonl \
            +task.query_emb_output_path=$CTX_EMBEDDINGS_DIR/query_reps_$dataset.pkl \
            +hydra.launcher.name=spar_generate_query_emb &
done

# after the embeddings are generated, run the retrieval script to generate retrieval results on the dev set (for tuning the concatenation weight)
DATASET_PATHS=""
QUERY_EMB_PATHS=""
OUTPUT_PATHS=""
for dataset in nq_dev squad1_dev webq_dev trivia_dev trec_dev
do
        DATASET_PATHS+=" <PATH_TO_DPR_DATA>/$dataset.jsonl"
        QUERY_EMB_PATHS+=" ${CTX_EMBEDDINGS_DIR}/query_reps_$dataset.pkl"
        OUTPUT_PATHS+=" $CTX_EMBEDDINGS_DIR/$dataset.json"
done

PYTHONPATH=.:$PYTHONPATH python dpr_scale/run_retrieval_multiset.py \
    --ctx_embeddings_dir $CTX_EMBEDDINGS_DIR \
    --passages_tsv_path <PATH_TO_DPR_PASSAGES>/psgs_w100.tsv \
    --questions_jsonl_paths $DATASET_PATHS \
    --query_emb_paths $QUERY_EMB_PATHS \
    --output_json_paths $OUTPUT_PATHS
```


### Tune the SPAR Concatenation Weight
With all embeddings and JSON results files generated for both the base model and Λ, run the following script to tune the concatenation weights for SPAR.

By default, it tunes weights on the dev sets of all five ODQA datasets (NQ, TriviaQA, SQuAD, WebQ, TREC).

```bash
# make sure you're in the root folder of the dpr-scale repo

PYTHONPATH=.:$PYTHONPATH python spar/spar_weight_tuning.py \
    --model_1_emb_dir ./spar/outputs/dpr_multiset/ \
    --model_2_emb_dir ./spar/outputs/wiki_lambda_bm25_bert_base/ \
    --output_dir ./spar/outputs/spar_odqa_weight_tuning/ 
```

For `DPR-multiset` and `Wiki BM25 Λ`, the selected weights are:
* NQ: 0.7
* SQuAD: 1.43
* TriviaQA: 0.9
* WebQ: 1.25
* TREC: 0.7

For MS MARCO, the selected weights are:
* SPAR (RocketQA + Marco BM25 Λ): 0.1
* SPAR (RocketQA + Marco UniCOIL Λ): 0.5

For BEIR experiments, the best weights are tuned on MS MARCO dev:

* SPAR (Contriever + Marco BM25 Λ): 0.006
* SPAR (Contriever + Marco UniCOIL Λ): 0.0333

* SPAR (GTR + Marco BM25 Λ): 0.001
* SPAR (GTR + Marco UniCOIL Λ): 0.007

(For certain MS MARCO models, much smaller weights are needed. In the standard weight tuning process, we select the weights from 0.1 to 10. But if the best weight lies on either end of the spectrum, we'll continue the searching by lowering (or increasing) the weights by 100x. For instance, if weight=0.1 gave the best results for a model, we'll do another grid search on [0.001, 0.1].)

### Run SPAR Retrieval

After tuning the concatenation weights, run the following script to perform SPAR retrieval with the selected weights on all five ODQA datasets:

```bash
# make sure you're in the root folder of the dpr-scale repo

PYTHONPATH=.:$PYTHONPATH python spar/spar_retrieval.py \
    --jsonl_dataset_paths <PATH_TO_DPR_DATA>/nq_test.jsonl <PATH_TO_DPR_DATA>/squad1_test.jsonl <PATH_TO_DPR_DATA>/trivia_test.jsonl <PATH_TO_DPR_DATA>/webq_test.jsonl <PATH_TO_DPR_DATA>/trec_test.jsonl \
    --model_1_emb_dir ./spar/outputs/dpr_multiset/ \
    --model_2_emb_dir ./spar/outputs/wiki_lambda_bm25_bert_base/ \
    --tsv_passages_path <PATH_TO_DPR_PASSAGES>/psgs_w100.tsv \
    --output_dir ./spar/outputs/spar_odqa_dpr_multiset_plus_wiki_lambda_bm25 \
    --weights 0.7 1.43 0.9 1.25 0.7
```

# License
SPAR is CC-BY-NC 4.0 licensed as of now.
