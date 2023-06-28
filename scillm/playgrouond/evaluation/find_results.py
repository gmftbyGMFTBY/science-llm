import json
import pprint
import ipdb

chatgpt = [json.loads(line) for line in open('chatgpt.txt').readlines()]
llama = [json.loads(line) for line in open('llama_gt.txt').readlines()]
chatglm = [json.loads(line) for line in open('chatglm_result.txt').readlines()]
vicuna = [json.loads(line) for line in open('vicuna_result.txt').readlines()]

dataset = []
for a, b, c, d in zip(chatgpt, llama, chatglm, vicuna):
    assert a['question'] == b['question'] == c['question'] == d['question']
    dataset.append({
        'question': a['question'],
        'reference': a['answer'],
        'chatgpt_answer': a['chatgpt_generation'],
        'chatglm_answer': c['generation'],
        'answer': b['generation'],
        'vicuna': d['generation'],
        'evidence': a['evidence']
    })

with open('ccc.json', 'w') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)
exit()

for sample in dataset:
    pprint.pprint(sample)
    ipdb.set_trace()

