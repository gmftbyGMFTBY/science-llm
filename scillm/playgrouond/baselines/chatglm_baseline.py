from transformers import AutoTokenizer, AutoModel
from copy import deepcopy
from tqdm import tqdm
import ipdb
import torch
import json

def main():
    args = {
        'data_path': 'processed_scimrc_test_set.json',
        'result_path': 'chatglm_scimrc_result_recall.txt',
        'recall': 'True'
    }
    model = AutoModel.from_pretrained(
        pretrained_model_name_or_path='THUDM/chatglm-6b',
        trust_remote_code=True
    ).half().cuda()
    tokenizer = AutoTokenizer.from_pretrained('THUDM/chatglm-6b', trust_remote_code=True)
    # load test dataset
    with open(args['data_path']) as f:
        question_set = set()
        data = []
        for sample in json.load(f):
            if sample['question'] not in question_set:
                question_set.add(sample['question'])
                data.append(sample)

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
            prompt = f'### Evidence:\n{evidence}\n\n### Question:\n{q}\n\n### Please generate the answer for the instruction based on the evidence.' 
            try:
                generation, history = model.chat(tokenizer, prompt, history=[], do_sample=False, max_length=2048)
            except:
                continue

            result_item = deepcopy(item)
            result_item['generation'] = generation
            f.write(json.dumps(result_item) + '\n')
            f.flush()

if __name__ == "__main__":
    main()
