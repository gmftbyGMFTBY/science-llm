import sys
import torch
from tqdm import tqdm
from itertools import chain
import json
import ipdb
sys.path.append('../scidpr')
from model import Agent, SciDPR 

if __name__ == "__main__":
    # with open('scimrc_test.json') as f:
    #     paper_data = json.load(f)
    with open('smrc_test.jsonl') as f:
        paper_data = [json.loads(line) for line in f.readlines()]
    with open('scimrc_test_sft.json') as f:
        data = json.load(f)
    args = {
        'question_model': '/home/lt/scidpr/question_encoder',
        'answer_model': '/home/lt/scidpr/answer_encoder',
        'q_max_len': 128,
        'a_max_len': 256,
        'mode': 'test',
        'topk': 2
    }
    model = SciDPR(**args)
    agent = Agent(model, args) 
    print(f'[!] ========== load the model and agent over ==========')

    # begin to process
    results = []
    for sample in tqdm(data):
        paper = paper_data[sample['paper_index']]
        # process all the sentences in paper
        sentences = [para['paragraphs'] for para in paper['full_text']]
        sentences = list(chain(*sentences))
        embeddings = agent.get_embedding(sentences, question=False, inner_bsz=64)
        query_embedding = agent.get_embedding([sample['question']])
        embeddings = torch.from_numpy(embeddings)
        query_embedding = torch.from_numpy(query_embedding)
        matrix = torch.matmul(query_embedding, embeddings.T)    # [1, N]
        index = matrix.topk(args['topk'], dim=-1)[1].tolist()[0]
        selected_sentences = [sentences[i] for i in index]

        sample['recall_evidence'] = selected_sentences
        results.append(sample)

    with open('processed_scimrc_test_set.json', 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


