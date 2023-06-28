import time
from copy import deepcopy
from collections import Counter
import openai
import logging
import json
import ipdb
import os
import random
import re
import string
from functools import partial
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
import evaluate


### QASPER Answer F1
def normalize_answer(s):
    """
    Taken from the official evaluation script for v1.1 of the SQuAD dataset.
    Lower text and remove punctuation, articles and extra whitespace.
    """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def token_f1_score(prediction, ground_truth):
    """
    Taken from the official evaluation script for v1.1 of the SQuAD dataset.
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def evaluate_answer_f1(path):
    with open(path) as f:
        results = [json.loads(line) for line in f.readlines()]
        results = [item for idx, item in enumerate(results) if idx not in yes_or_no]

    max_f1, max_precision, max_recall = [], [], []
    for sample in results:
        pause_f1, pause_precision, pause_recall = [], [], []
        for reference in sample['answer']:
            p, r, f1 = token_f1_score(sample['generation'].strip(), reference)
            pause_recall.append(r)
            pause_f1.append(f1)
            pause_precision.append(p)
            max_f1.append(max(pause_f1))
            max_precision.append(max(pause_precision))
            max_recall.append(max(pause_recall))
    print(f'[!] answer F1: {round(np.mean(max_f1), 4)}')
    print(f'[!] answer Precision: {round(np.mean(max_precision), 4)}')
    print(f'[!] answer Recall: {round(np.mean(max_recall), 4)}')

def evaluate_rouge(path, rouge):
    with open(path) as f:
        results = [json.loads(line) for line in f.readlines()]
        results = [item for idx, item in enumerate(results) if idx not in yes_or_no]
    scores = []
    r = rouge.compute(
        predictions=[sample['generation'].strip() for sample in results], 
        references=[sample['answer'][0] for sample in results]
    )
    print(r)


def evaluate_bertscore(path, rouge):
    with open(path) as f:
        results = [json.loads(line) for line in f.readlines()]
        results = [item for idx, item in enumerate(results) if idx not in yes_or_no]
    scores = []
    r = bertscore.compute(
        predictions=[sample['generation'].strip() for sample in results], 
        references=[sample['answer'][0] for sample in results],
        lang='en'
    )
    r = round(np.mean(r['f1']), 4)
    print('BERTScore:', r)

def evaluate_bleu(path):
    with open(path) as f:
        results = [json.loads(line) for line in f.readlines()]
        results = [item for idx, item in enumerate(results) if idx not in yes_or_no]
    scores = []
    for sample in tqdm(results):
        reference = [i.split() for i in sample['answer']]
        generation = sample['generation'].strip().split()
        s = sentence_bleu(reference, generation)
        scores.append(s)
    print('BLEU:', round(np.mean(scores), 4))

if __name__ == "__main__":

    # yes or no index
    with open('llama_gt.txt') as f:
        yes_or_no = []
        for idx, line in enumerate(f.readlines()):
            if json.loads(line)['yes_no']:
                yes_or_no.append(idx)
        yes_or_no = set(yes_or_no)

    rouge = evaluate.load('rouge')
    bertscore = evaluate.load('bertscore')
    print(f'[!] load rouge and bertscore over')
    # path = 'ours_llama_gt.txt'
    # path = 'ours_llama_recall.txt'
    path = 'llama_qasper_gt.txt'
    evaluate_answer_f1(path)
    evaluate_bleu(path)
    evaluate_rouge(path, rouge)
    evaluate_bertscore(path, bertscore)
    print('=' * 30)
