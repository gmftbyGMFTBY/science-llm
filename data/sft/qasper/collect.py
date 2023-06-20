import json
from tqdm import tqdm
import ipdb

def remove_duplicate(d):
    question_set = set()
    dataset = {}
    counter = 0
    for sample in d:
        if sample['question'] not in question_set:
            question_set.add(sample['question'])
            sample['answer'] = [sample['answer']]
            dataset[sample['question']] = sample
        else:
            dataset[sample['question']]['answer'].append(sample['answer'])
            counter += 1
    print(f'[!] remove {counter} samples and save {len(dataset)} samples')
    dataset = [dataset[key] for key in dataset]
    return dataset

if __name__ == "__main__":
    yes_no_datasets, datasets = [], []
    for mode in ['train', 'dev', 'test']:
        data = json.load(open(f'qasper-{mode}-v0.3.json'))
        yes_no_dataset, dataset = [], []
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
                            dataset.append({
                                'question': question,
                                'answer': answer,
                                'evidence': '\n'.join(evidence),
                                'yes_no': False
                            })
                        elif a['extractive_spans']:
                            answer = ''.join([f'{idx}. {i}\n' for idx, i in enumerate(a['extractive_spans'])])
                            dataset.append({
                                'question': question,
                                'answer': answer,
                                'evidence': '\n'.join(evidence),
                                'yes_no': False
                            })
                        else:
                            if a['yes_no'] is not None and evidence:
                                if a['yes_no'] is True:
                                    answer = 1
                                else:
                                    answer = 0
                                yes_no_dataset.append({
                                    'question': question,
                                    'answer': answer,
                                    'evidence': '\n'.join(evidence),
                                    'yes_no': True
                                })
                                dataset.append({
                                    'question': question,
                                    'answer': 'Yes.' if answer else 'No.',
                                    'evidence': '\n'.join(evidence),
                                    'yes_no': True
                                })
                            else:
                                # raise Exception(f'[!] something wrong')
                                pass

        print(f'[!] collect {len(dataset)} samples')
        datasets.append(dataset)
        yes_no_datasets.append(yes_no_dataset)
    train_dataset = datasets[0] + datasets[1]
    test_dataset = datasets[2]
    json.dump(train_dataset, open(f'qasper_train_sft.json', 'w'), indent=4, ensure_ascii=False)
    test_dataset = remove_duplicate(test_dataset)
    json.dump(test_dataset, open(f'qasper_test_sft.json', 'w'), indent=4, ensure_ascii=False)

    train_dataset = yes_no_datasets[0] + yes_no_datasets[1]
    test_dataset = yes_no_datasets[2]
    json.dump(train_dataset, open(f'qasper_yes_no_train_sft.json', 'w'), indent=4, ensure_ascii=False)
    json.dump(test_dataset, open(f'qasper_yes_no_test_sft.json', 'w'), indent=4, ensure_ascii=False)
