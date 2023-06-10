from datasets import load_dataset
import json
from transformers import LlamaTokenizer
from tqdm import tqdm
import pprint
import numpy as np
from tqdm import tqdm
import ipdb

def download():
    dataset = load_dataset('togethercomputer/RedPajama-Data-1T', split='train', streaming=True)
    print(f'[!] initialize the redpajama dataset over')

    tokenizer = LlamaTokenizer.from_pretrained('/home/johnlan/pretrained_models/LLaMA-7B-HF')
    f = open('redpajama.json', 'w')
    f_token = open('redpajama_tokens.json', 'w')

    # commoncrawl vs arxiv with 1:19
    max_arxiv_tokens, max_cc_tokens = int(3.8e9), int(2e8)
    current_arxiv_tokens, current_cc_tokens = 0, 0
    current_arxiv_item, current_cc_item = 0, 0
    pbar_arxiv = tqdm(total=max_arxiv_tokens)
    pbar_cc = tqdm(total=max_cc_tokens)

    for item in dataset:
        if item['red_pajama_subset'] in ['arxiv', 'commoncrawl']:
            tokens = tokenizer.encode(item['text'], add_special_tokens=False)
            if item['red_pajama_subset'] == 'arxiv':
                current_arxiv_tokens += len(tokens)
                index_in_raw_file = current_arxiv_item
                current_arxiv_item += 1
                pbar_arxiv.update(len(tokens))
            else:
                current_cc_tokens += len(tokens)
                index_in_raw_file = current_cc_item
                current_cc_item += 1
                pbar_cc.update(len(tokens))
            token_item = {
                'tokens': tokens,
                'red_pajama_subset': item['red_pajama_subset'],
                'index_in_raw_file': index_in_raw_file
            }
            f.write(json.dumps(item) + '\n')
            f_token.write(json.dumps(token_item) + '\n')
            f.flush()
            f_token.flush()

if __name__ == "__main__":
    download()
