from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig, LlamaTokenizer, StoppingCriteria, StoppingCriteriaList
from peft import prepare_model_for_kbit_training
from copy import deepcopy
import os
from tqdm import tqdm
import ipdb
import torch
import json

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = stops
        self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        stop_count = 0
        for stop in self.stops:
            stop_count = (stop == input_ids[0]).sum().item()
        if stop_count >= self.ENCOUNTERS:
            return True
        return False

def main():
    args = {
        'data_path': 'processed_scimrc_test_set.json',
        'result_path': 'vicuna_scimrc_result_recall.txt',
        'recall': 'True'
    }
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path='/home/lt/vicuna_model',
    ).half().cuda()
    tokenizer = LlamaTokenizer.from_pretrained('/home/lt/vicuna_model')
    # load test dataset
    with open(args['data_path']) as f:
        question_set = set()
        data = []
        for sample in json.load(f):
            if sample['question'] not in question_set:
                question_set.add(sample['question'])
                data.append(sample)
    if os.path.exists(args['result_path']):
        with open(args['result_path']) as f:
            number = len(f.readlines())
            data = data[number:]

    with torch.no_grad():
        results = []
        f = open(args['result_path'], 'a')
        for item in tqdm(data):
            q = item['question']
            if args['recall'] == 'True':
                evidence = '\n'.join(item['recall_evidence'])
            elif args['recall'] == 'False':
                evidence = item['evidence']
            else:
                raise Exception(f'[!] Unknown recall mode: {args["recall"]}')
            prompt = f'### Human: please generate the answer for the instruction based on the evidence. 1. Evidence:\n{evidence}\n\n2. Question:\n{q}\n### Assistant:' 
            tokens = torch.LongTensor(tokenizer.encode(prompt, add_special_tokens=False)).unsqueeze(0).cuda()
            length = len(tokens[0])
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=[13], encounters=1)])
            try:
                output = model.generate(tokens, do_sample=False, max_new_tokens=128)
            except:
                continue
            generation = tokenizer.decode(output[0, length:], skip_special_tokens=False)
            if '###' in generation:
                generation = generation[:generation.index('###')].strip()
            result_item = deepcopy(item)
            result_item['generation'] = generation
            f.write(json.dumps(result_item) + '\n')
            f.flush()

if __name__ == "__main__":
    main()
