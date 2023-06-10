import copy
import linecache
from itertools import chain,islice
from copy import deepcopy
import os
import json
from tqdm import tqdm
import ipdb
import random
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass, field
from typing import Callable, Dict, Sequence

import torch
import torch.distributed as dist
import transformers
from torch.utils.data import Dataset


def iter_count(file_name):
    from itertools import (takewhile, repeat)
    buffer = 1024 * 1024
    with open(file_name) as f:
        buf_gen = takewhile(lambda x: x, (f.read(buffer) for _ in repeat(None)))
        return sum(buf.count('\n') for buf in buf_gen)


class PretrainDataset(Dataset):

    def __init__(self, **args):
        super(PretrainDataset, self).__init__()
        self.args = args
        self.tokenizer = args['tokenizer'] 
        # cached dataset
        self.cache_f_reader = open(self.args['data_path'])
        self.cache_tokens = []
        self.instance_num = iter_count(args['data_path'])
        print(f'[!] collect {self.instance_num} samples from {args["data_path"]}')

    def __len__(self):
        # useless thing
        return 64

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if len(self.cache_tokens) < self.args['max_seq_length']:
            cache = []
            for _ in range(self.args['max_dataset_cache_size']):
                line = self.cache_f_reader.readline().strip()
                if line:
                    cache.append(json.loads(line))
                else:
                    print(f'[!] read out of the file, reload ...')
                    self.cache_f_reader = open(self.args['data_path'])
            random.shuffle(cache)
            # concatentate
            self.cache_tokens = []
            for item in cache:
                self.cache_tokens += item['tokens'] + [self.tokenizer.eos_token_id]
        tokens = deepcopy(self.cache_tokens[:self.args['max_seq_length']])
        del self.cache_tokens[:self.args['max_seq_length']]
        return torch.LongTensor(tokens)

    def collate(self, instances):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            instances,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(instances, batch_first=True, padding_value=-100)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

class PretrainTestDataset(Dataset):

    def __init__(self, **args):
        super(PretrainTestDataset, self).__init__()
        self.args = args
        self.tokenizer = args['tokenizer'] 
        # cached dataset
        self.cache_f_reader = open(self.args['data_path'])
        self.cache = list(islice(self.cache_f_reader, 0, 1000))
        self.cache = [json.loads(i) for i in self.cache]
        self.cache_tokens = []
        for item in self.cache:
            self.cache_tokens += item['tokens'] + [self.tokenizer.eos_token_id]
        self.tokens = [self.cache_tokens[i:i+self.args['max_seq_length']]for i in range(0, len(self.cache_tokens), self.args['max_seq_length'])]
        self.tokens = self.tokens[:100]
        print(f'[!] load {len(self.tokens)} samples for testing')

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return torch.LongTensor(self.tokens[i])

    def collate(self, instances):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            instances,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(instances, batch_first=True, padding_value=-100)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

if __name__ == "__main__":
    from transformers import LlamaTokenizer
    from tqdm import tqdm
    args = {'max_seq_length': 4096, 'tokenizer': LlamaTokenizer.from_pretrained('decapoda-research/llama-7b-hf'), 'max_dataset_cache_size': 100, 'data_path': '../data/pretrain/train/split_00'}
    dataset = PretrainDataset(**args)
    for i in tqdm(range(50000)):
        dataset[i]

