# Synthetic Query Generation

> Data augmentation script for Section 3.1.2 Synthetic Queries from Llama-3.3-70B-Instruct

This script generates synthetic queries from a text corpus using a `meta-llama/Llama-3.3-70B-Instruct` with vllm. Each output consists of a task description, a query, and the query language, aligned with the document text.


## Usage

Using Wikipedia corpus as example.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python generate_queries.py \
  --shard 0 \
  --num_shards 1000 \
  --output_dir ./output \
  --dataset_name miracl/miracl-corpus \
  --dataset_config en \
  --dataset_split train
```

## Input/Output Example
Example Input Document `(data['text'])`

```txt
纳米技术是一种新兴的平台，能够帮助我们测量、理解和操纵干细胞。例如，磁性纳米颗粒和量子点可以用于标记和追踪干细胞；纳米颗粒、碳纳米管和聚合物复合体可以用于将基因/寡核苷酸和蛋白质/肽类递送到细胞内；而工程化的纳米尺寸支架可以用于干细胞分化和移植。这篇评论文章讨论了纳米技术在干细胞追踪、分化和移植中的应用，并进一步探讨了其实用性和潜在的细胞毒性问题。
```

Generated Output JSONL Entry

```json
{
  "query_id": "doc_456",
  "task": "Retrieve a scientific paper abstract to verify a claim.",
  "query": "0维生物材料缺乏诱导性质",
  "language": "Chinese"
}
```


In this script, we generate both the **retrieval task type** and the **query text** for each input document. This encourages the model to synthesize diverse retrieval intents (e.g., fact verification, QA, scientific search) across multiple languages and domains.

However, **only the query part is used for downstream retrieval training**, without the task description. The goal is for the retriever to learn the **query’s implicitly intent** through the diversity of queries generated, without being explicitly told the task. This promotes generalization across different retrieval task formulations.
