import json
import os
from transformers import LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import pprint
import numpy as np
from tqdm import tqdm
import ipdb

def process():
    tokenizer = AutoTokenizer.from_pretrained('baichuan-inc/baichuan-7B', trust_remote_code=True)
    with open('redpajama_test.json') as f:
        dataset = []
        for line in f.readlines():
            dataset.append(json.loads(line)['text'])
        dataset = dataset[:100]
    counting = 0
    with open('redpajama_tokens_test_v1_chinese.json', 'w') as f:
        pbar = tqdm(dataset)
        for sample in pbar:
            tokens = tokenizer.encode(sample, add_special_tokens=False)
            f.write(json.dumps({'tokens': tokens}, ensure_ascii=False) + '\n')
            counting += len(tokens)
            pbar.set_description(f'[!] collecting {counting} tokens')

if __name__ == "__main__":
    process()
