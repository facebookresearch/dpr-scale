# Synthetic Query Generation

> Passage reranking script for Section 3.2.3 Listwise Reranking from Llama-3.3-70B-Instruct

This script generates synthetic queries from a text corpus using a `meta-llama/Llama-3.3-70B-Instruct` with vllm. Each output consists of a task description, a query, and the query language, aligned with the document text.


## Usage

Using a `.jsonl` input file containing queries and passages:


```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python rerank_from_jsonl.py \
  --input_path ./input/top20.jsonl \
  --output_path ./output/reranked.jsonl
```

## Input/Output Example
Example Input JSONL Entry

```json
{
  "query_id": "q123",
  "query": "What is the capital of France?",
  "passages": [
    {"docid": "d1", "text": "Berlin is the capital of Germany."},
    {"docid": "d2", "text": "Paris is the capital and most populous city of France."},
    ...
  ]
}
```

Generated Output JSONL Entry

```json
{
  "query_id": "q123",
  "rerank_raw": "[2] > [5] > [1] > ...",
  "passage_ids": ["d1", "d2", ..., "d20"]
}
```
