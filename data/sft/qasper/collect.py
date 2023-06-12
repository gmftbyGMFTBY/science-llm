import json
from tqdm import tqdm
import ipdb

if __name__ == "__main__":
    datasets = []
    for mode in ['train', 'dev', 'test']:
        data = json.load(open(f'qasper-{mode}-v0.3.json'))
        dataset = []
        for paper_id in tqdm(data):
            paper = data[paper_id]
            for qa in paper['qas']:
                question = qa['question']
                for a in qa['answers']:
                    a = a['answer']
                    if a['unanswerable'] is False:
                        evidence = a['evidence'] + a['highlighted_evidence']
                        if a['free_form_answer']:
                            answer = a['free_form_answer']
                        elif a['extractive_spans']:
                            answer = ''.join([f'{idx}. {i}\n' for idx, i in enumerate(a['extractive_spans'])])
                        else:
                            if a['yes_no'] is not None:
                                if a['yes_no'] is True:
                                    answer = 'Yes.'
                                else:
                                    answer = 'No.'
                            else:
                                raise Exception(f'[!] something wrong')
                        dataset.append({
                            'question': question,
                            'answer': answer,
                            'evidence': '\n'.join(evidence)
                        })

        print(f'[!] collect {len(dataset)} samples')
        datasets.append(dataset)
    train_dataset = datasets[0] + datasets[1]
    test_dataset = datasets[2]
    json.dump(train_dataset, open(f'qasper_train_sft.json', 'w'), indent=4, ensure_ascii=False)
    json.dump(test_dataset, open(f'qasper_test_sft.json', 'w'), indent=4, ensure_ascii=False)
