from datasets import load_dataset
import json
import ipdb

dataset = load_dataset("databricks/databricks-dolly-15k")['train']

data = []

for sample in dataset:
    data.append({
        'question': sample['instruction'],
        'evidence': sample['context'],
        'answer': sample['response'],
        'yes_no': False
    })

with open('train.json', 'w') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
