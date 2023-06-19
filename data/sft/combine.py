import json
import random

random.seed(0)

with open('qasper/qasper_train_sft.json') as f:
    data = json.load(f)

with open('scimrc/scimrc_train.json') as f:
    data.extend(json.load(f))

random.shuffle(data)

with open('train.json', 'w') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
    print(f'[!] collect {len(data)} samples')
