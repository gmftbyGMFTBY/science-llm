import copy
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
from tqdm import tqdm

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    max_length: str,
    bos_token_id,
    eos_token_id
) -> Dict:
    input_ids, labels = [], []
    for s, t in zip(sources, targets):
        s_tokens = tokenizer.encode(s, add_special_tokens=False)
        t_tokens = tokenizer.encode(t, add_special_tokens=False)
        inpt = s_tokens + t_tokens + [eos_token_id]
        label = [-100] * len(s_tokens) + t_tokens + [eos_token_id]
        inpt = inpt[-max_length:]
        label = label[-max_length:]
        input_ids.append(torch.LongTensor(inpt))
        labels.append(torch.LongTensor(label))
    return dict(input_ids=input_ids, labels=labels)


class QASPERDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, **args):
        super(QASPERDataset, self).__init__()
        self.args = args
        list_data_dict = json.load(open(args['train_data_path']))
        self.tokenizer = args['tokenizer']

        prompt_input = '### Evidence:\n{evidence}\n\n### Instruction:\n{question}\n\n### Response:'
        sources = [prompt_input.format_map(example) for example in tqdm(list_data_dict)]
        targets = [example['answer'] for example in list_data_dict]
        data_dict = preprocess(sources, targets, self.tokenizer, self.args['max_seq_length'], self.tokenizer.bos_token_id, self.tokenizer.eos_token_id)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        print(f'[!] collect {len(self.input_ids)} samples for training')

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    def collate(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        lengths = [len(item) for item in input_ids]
        max_length = max(lengths)
        attention_mask = torch.LongTensor(
           [[1] * length + [0] * (max_length - length) for length in lengths]
        )
        attention_mask = attention_mask.to(torch.bool)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=0
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask
        )

class EmotionalDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, **args):
        super(EmotionalDataset, self).__init__()
        self.args = args
        list_data_dict = json.load(open(args['train_data_path']))
        self.tokenizer = args['tokenizer']
        self.data = []
        for sample in list_data_dict:
            string = self.tokenizer.eos_token.join(sample)
            tokens = self.tokenizer.encode(string, add_special_tokens=False)
            self.data.append(torch.LongTensor(tokens))
        print(f'[!] collect {len(self.data)} samples for training')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.data[i], labels=self.data[i])

    def collate(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        lengths = [len(item) for item in input_ids]
        max_length = max(lengths)
        attention_mask = torch.LongTensor(
           [[1] * length + [0] * (max_length - length) for length in lengths]
        )
        attention_mask = attention_mask.to(torch.bool)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.eos_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask
        )
