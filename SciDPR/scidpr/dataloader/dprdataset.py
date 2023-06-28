from header import *
from collections import defaultdict
from datasets import load_dataset, load_from_disk
from .utils import *


class DPRDataset(Dataset):

    def __init__(self, path, **args):
        self.args = args
        self.question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(self.args['question_model'])
        self.answer_tokenizer = DPRContextEncoderTokenizer.from_pretrained(self.args['answer_model'])
        data = json.load(open(path))
        self.data = []
        for item in data:
            for a in item['a']:
                self.data.append({'q': item['q'], 'a': a, 'paper_id': item['paper_id']})
        
        # original paper data
        path_train = path.replace('train.json', 'qasper-train-v0.3.json')
        path_dev = path.replace('train.json', 'qasper-dev-v0.3.json')
        original_data = json.load(open(path_train))
        original_data.update(json.load(open(path_dev)))
        self.paper_data = original_data
        print(f'[!] load {len(original_data)} papers and {len(self.data)} training samples')
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        q = self.data[i]['q']
        a = self.data[i]['a']
        paper = [para['paragraphs'] for para in self.paper_data[self.data[i]['paper_id']]['full_text']]
        paper = list(chain(*paper))
        if len(paper) > self.args['negative_num']:
            na = random.sample(paper, self.args['negative_num'])
        else:
            na = paper
        q_ids = [self.question_tokenizer.cls_token_id] + self.question_tokenizer.encode(q, add_special_tokens=False)[:self.args['q_max_len']-2] + [self.question_tokenizer.sep_token_id]
        a_ids = [self.answer_tokenizer.cls_token_id] + self.answer_tokenizer.encode(a, add_special_tokens=False)[:self.args['a_max_len']-2] + [self.answer_tokenizer.sep_token_id]
        na_ids = [[self.answer_tokenizer.cls_token_id] + item[:self.args['a_max_len']-2] + [self.answer_tokenizer.sep_token_id] for item in self.answer_tokenizer.batch_encode_plus(na, add_special_tokens=False)['input_ids']]
        return q_ids, [a_ids] + na_ids

    def collate(self, batch):
        q_ids = [torch.LongTensor(i) for i, j in batch]
        a_ids = [[torch.LongTensor(k) for k in j] for i, j in batch]
        a_ids = list(chain(*a_ids))
        size = [len(i) for _, i in batch]
        labels = [0] + np.cumsum(size)[:-1].tolist()
        
        q_ids = pad_sequence(q_ids, batch_first=True, padding_value=self.question_tokenizer.pad_token_id)
        a_ids = pad_sequence(a_ids, batch_first=True, padding_value=self.answer_tokenizer.pad_token_id)
        q_mask = generate_mask(q_ids, pad_token_idx=self.question_tokenizer.pad_token_id)
        a_mask = generate_mask(a_ids, pad_token_idx=self.answer_tokenizer.pad_token_id)
        labels = torch.LongTensor(labels)
        return {
            'q_ids': q_ids,
            'a_ids': a_ids,
            'q_mask': q_mask,
            'a_mask': a_mask,
            'labels': labels
        }

if __name__ == "__main__":
    pass
