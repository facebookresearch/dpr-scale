import json
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Generate synthetic queries for retrieval tasks')
parser.add_argument('--shard', type=int, required=True, help='Shard index for the output file')
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--dataset_name', type=str, default='miracl/miracl-corpus', help='Name of the dataset to load')
parser.add_argument('--dataset_config', type=str, default=None, help='Configuration of the dataset to load')
parser.add_argument('--dataset_split', type=str, default='train', help='Split of the dataset to load')
parser.add_argument('--num_shards', type=int, default=1000, help='Total number of shards to split the dataset into')

args = parser.parse_args()

corpus = load_dataset(args.dataset_name, args.dataset_config)[args.dataset_split].shard(num_shards=args.num_shards, index=args.shard, contiguous=False)

llm = LLM(model="meta-llama/Llama-3.3-70B-Instruct", tensor_parallel_size=4, enable_prefix_caching=True)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")
sampling_params = SamplingParams(temperature=0.8, top_p=0.9, max_tokens=256, stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")])

def format_massage(text):
    text = text.replace('\n', '').strip()
    messages = [
        {"role": "system", "content": "You are a search agent. Given a text, your task is to brainstorm a retrieval task this text can support. (e.g. QA, websearch, fact verification). The task should be written in English, but the query should be in the same language as the document. Be creative."},
        {'role': "user", "content": "Text: Cytoplasm is the fluid where the cellular organelles are suspended. It fills up the spaces that are not occupied by the organelles. The constituents of cytoplasm are cytosol, organelles and cytoplasmic inclusions."},
        {'role': "assistant", "content": "Task: Given a web search query, retrieve a relevant passage to answer.\nQuery: cytoplasmic organelles definition\nLanguage: English"},
        {'role': "user", "content": "Text: চেঙ্গিজ খান (মঙ্গোলীয়: Чингис Хаан আ-ধ্ব-ব: [ʧiŋgɪs χaːŋ], ), (১১৬২[1]–আগস্ট ১৮, ১২২৭) প্রধান মঙ্গোল রাজনৈতিক ও সামরিক নেতা বা মহান খান, ইতিহাসেও তিনি অন্যতম বিখ্যাত সেনাধ্যক্ষ ও সেনাপতি। জন্মসূত্রে তার নাম ছিল তেমুজিন (মঙ্গোলীয়: Тэмүжин )। তিনি মঙ্গোল গোষ্ঠীগুলোকে একত্রিত করে মঙ্গোল সাম্রাজ্যের (Екэ Монгол Улус; ১২০৬ - ১৩৬৮) গোড়াপত্তন করেন। নিকট ইতিহাসে এটিই ছিল পৃথিবীর সর্ববৃহৎ সম্রাজ্য। তিনি মঙ্গোলিয়ার বোরজিগিন বংশে জন্ম নিয়েছিলেন। এক সাধারণ গোত্রপতি থেকে নিজ নেতৃত্বগুণে বিশাল সেনাবাহিনী তৈরি করেন।যদিও বিশ্বের কিছু অঞ্চলে চেঙ্গিজ খান অতি নির্মম ও রক্তপিপাসু বিজেতা হিসেবে চিহ্নিত[2] তথাপি মঙ্গোলিয়ায় তিনি বিশিষ্ট ব্যক্তি হিসেবে সম্মানিত ও সকলের ভালোবাসার পাত্র। তাকে মঙ্গোল জাতির পিতা বলা হয়ে থাকে। একজন খান হিসেবে অধিষ্ঠিত হওয়ার পূর্বে চেঙ্গিজ পূর্ব ও মধ্য এশিয়ার অনেকগুলো যাযাবর জাতিগোষ্ঠীকে একটি সাধারণ সামাজিক পরিচয়ের অধীনে একত্রিত করেন। এই সামাজিক পরিচয়টি ছিল মঙ্গোল।"},
        {'role': "assistant", "content": "Task: Find a Wikipedia article to answer the question.\nQuery: চেঙ্গিস খান কোন বংশের রাজা ছিলেন ?\nLanguage: Bengali"},
        {'role': "user", "content": "Text: 纳米技术是一种新兴的平台，能够帮助我们测量、理解和操纵干细胞。例如，磁性纳米颗粒和量子点可以用于标记和追踪干细胞；纳米颗粒、碳纳米管和聚合物复合体可以用于将基因/寡核苷酸和蛋白质/肽类递送到细胞内；而工程化的纳米尺寸支架可以用于干细胞分化和移植。这篇评论文章讨论了纳米技术在干细胞追踪、分化和移植中的应用，并进一步探讨了其实用性和潜在的细胞毒性问题。"},
        {'role': "assistant", "content": "Task: Retrieve a scientific paper abstract to verify a claim.\nQuery: 0维生物材料缺乏诱导性质\nLanguage: Chinese"},
        {'role': "user", "content": f"Text: {text}"},
    ]
    return messages

candidate_ids = []
candidate_inputs = []
for data in corpus:
    candidate_ids.append(data['docid'])
    text = data['text']
    messages = format_massage(text)
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    candidate_inputs.append(prompt)

outputs = llm.generate(candidate_inputs, sampling_params)

with open(f'{args.output_dir}/{args.shard}.jsonl', 'w', encoding="utf-8") as f:
    for id_, output in tqdm(zip(candidate_ids, outputs)):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        try:
            task, query = generated_text.split('\nQuery:')[:2]
            query, language = query.split('\nLanguage:')
            task = task.replace('Task:', '').strip()
            query = query.strip()
            language = language.strip().split('\n')[0].strip()
        except:
            continue
        f.write(json.dumps({
            'query_id': id_,
            'task': task,
            'query': query,
            'language': language,
        }, ensure_ascii=False)+'\n')