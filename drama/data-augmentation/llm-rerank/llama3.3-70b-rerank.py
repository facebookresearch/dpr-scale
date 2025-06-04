import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Listwise reranking from input JSONL with query and passages')
parser.add_argument('--input_path', type=str, required=True, help='Path to the input .jsonl file')
parser.add_argument('--output_path', type=str, required=True, help='Path to the output .jsonl file')
args = parser.parse_args()

llm = LLM(model="meta-llama/Llama-3.3-70B-Instruct", tensor_parallel_size=4)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")

sampling_params = SamplingParams(
    temperature=0,
    top_p=1,
    max_tokens=256,
    stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
)

def format_message(query, passages):
    user_message = f"I will provide you with 20 passages, each indicated by a numerical identifier []. Rank the passages based on their relevance to the search query: {query}.\n\n"
    for idx, passage in enumerate(passages):
        passage_text = passage['text'].replace('\n', ' ')
        passage_text = tokenizer.decode(tokenizer.encode(passage_text)[1:257])
        user_message += f"[{idx+1}] {passage_text}\n"
    user_message += f'\nQuery: {query}\n\n'
    user_message += 'Rank the 20 passages above based on their relevance to the search query. All the passages should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., [4] > [2]. Only respond with the ranking results, do not say any word or explain.'
    messages = [
        {"role": "system", "content": "You are a Search Agent, an intelligent assistant that can rank passages based on their relevancy to the query of a retrieval task."},
        {"role": "user", "content": user_message}
    ]
    return messages

# Read input
query_data = []
with open(args.input_path, 'r', encoding='utf-8') as f:
    for line in tqdm(f):
        data = json.loads(line)
        query_data.append(data)

# Prepare prompts
prompts = []
query_ids = []
all_passage_ids = []

for item in query_data:
    query_id = item['query_id']
    query = item['query']
    passages = item['passages'][:20]  # Ensure only top-20
    prompt = tokenizer.apply_chat_template(format_message(query, passages), add_generation_prompt=True, tokenize=False)
    prompts.append(prompt)
    query_ids.append(query_id)
    all_passage_ids.append([p['docid'] for p in passages])

# Generate responses
outputs = llm.generate(prompts, sampling_params)

# Write output
with open(args.output_path, 'w', encoding='utf-8') as f:
    for query_id, output, passage_ids in tqdm(zip(query_ids, outputs, all_passage_ids), total=len(query_ids)):
        generated_text = output.outputs[0].text
        f.write(json.dumps({
            'query_id': query_id,
            'rerank_raw': generated_text,
            'passage_ids': passage_ids
        }) + '\n')
