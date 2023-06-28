import time
from copy import deepcopy
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
from rouge_score import rouge_scorer


prompt = [
    {
        'role': "system", 'content': '''You are assistant to evaluate the performance of two answers for a question about some evidences. Specfically, given the evidence, question and multiple groud-truth answers, you need to decide which candidate answer is better. Generate the index of the best candidate or generate "NULL" if it is hard to decide. Note that return the result in json format, like:
{
    "result": "A"
}
or like:
{
    "result": "NULL"
}
        '''
    }
]


def openai_chat_completion(messages, model_name, sleep_time=240):
    while True:
        try:
            completion_batch = openai.ChatCompletion.create(model=model_name, messages=messages, temperature=0.2, top_p=1.0, max_tokens=128, n=1)
            content = completion_batch.choices[0]['message']['content']
            return {'text': content, 'finish_reason': completion_batch.choices[0]['finish_reason']}
        except openai.error.OpenAIError as e:
            if "Please reduce your prompt" in str(e):
                return None
            else:
                print('Hit request rate limit; retrying ...')
                time.sleep(sleep_time)


def encode_prompt(question, evidence, gt_answers, answer_a, answer_b):
    """Encode multiple prompt instructions into a single string."""
    global prompt
    prompt = deepcopy(prompt)
    gt_answers = '\n'.join(gt_answers)
    prompt.append({
        'role': 'user', 'content': f'### Evidence: {evidence}\n### Question: {question}\n### Ground-Truth:\n{gt_answers}\n### Candidate Answer:\nA. {answer_a}\nB. {answer_b}'    
    })
    return prompt


if __name__ == "__main__":
    # load chatgpt results
    with open('chatgpt.txt') as f:
        chatgpt_results = [json.loads(line) for line in f.readlines()][:100]
    with open('llama_gt.txt') as f:
        ours_results = [json.loads(line) for line in f.readlines()][:100]
    ipdb.set_trace()

    dataset = []
    for a, b in zip(chatgpt_results, ours_results):
        if a['question'] == b['question']:
            if random.random() < 0.5:
                aa = a['chatgpt_generation']
                bb = b['generation']
                index = 0
            else:
                ipdb.set_trace()
                aa = a['generation']
                bb = b['chatgpt_generation']
                index = 1

            dataset.append({
                'question': a['question'],
                'evidence': a['evidence'],
                'gt_answers': a['answer'],
                'answer_a': aa,
                'answer_b': bb,
                'chatgpt_index': index
            })

    if os.path.exists('generated_result.txt'):
        with open('generated_result.txt') as f:
            generated_samples = f.readlines()
            number = len(generated_samples)
    else:
        generated_samples = []
        number = 0
    print(f'[!] total sample: {len(dataset)}, load {number} samples, left {len(dataset) - number} samples to process')

    data = data[number:]
    with open('generated.txt', 'a') as f:
        for sample in tqdm(data):
            prompt = encode_prompt(sample['question'], sample['evidence'], sample['gt_answers'], sample['answer_a'], sample['answer_b'])
            result = openai_chat_completion(prompt, 'gpt-3.5-turbo', sleep_time=300)
            result_sample = deepcopy(sample)
            result_sample['chatgpt_generation'] = result['text']
            f.write(json.dumps(result_sample) + '\n')
            f.flush()
            time.sleep(30)
