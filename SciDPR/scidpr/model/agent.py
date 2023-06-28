from header import *
from dataloader import *

class Agent:
    
    def __init__(self, model, args):
        super(Agent, self).__init__()
        self.args = args
        self.model = model

        self.question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(self.args['question_model'])
        self.answer_tokenizer = DPRContextEncoderTokenizer.from_pretrained(self.args['answer_model'])

        if torch.cuda.is_available():
            self.model.cuda()
        if args['mode'] in ['train']:
            self.set_optimizer_scheduler_ddp()

    @torch.no_grad()
    def get_embedding(self, texts, question=True, inner_bsz=128):
        self.model.eval()

        max_len = self.args['q_max_len'] if question else self.args['a_max_len']
        pad_token_id = self.question_tokenizer.pad_token_id if question else self.answer_tokenizer.pad_token_id

        ids = []
        for text in texts:
            if question:
                ids_ = [self.question_tokenizer.cls_token_id] + self.question_tokenizer.encode(text, add_special_tokens=False)[:max_len-2] + [self.question_tokenizer.sep_token_id]
            else:
                ids_ = [self.answer_tokenizer.cls_token_id] + self.answer_tokenizer.encode(text, add_special_tokens=False)[:max_len-2] + [self.answer_tokenizer.sep_token_id]
            ids.append(ids_)

        reps = []
        for i in tqdm(range(0, len(ids), inner_bsz)):
            sub_ids = [torch.LongTensor(i) for i in ids[i:i+inner_bsz]]
            sub_ids = pad_sequence(sub_ids, batch_first=True, padding_value=pad_token_id)
            mask = generate_mask(sub_ids, pad_token_idx=pad_token_id)
            sub_ids, mask = sub_ids.cuda(), mask.cuda()
            if question:
                rep = self.model.get_question_embedding(sub_ids, mask)
            else:
                rep = self.model.get_answer_embedding(sub_ids, mask)
            reps.append(rep)
        reps = torch.cat(reps)    # [B, E]
        return reps.cpu().numpy()

    def train_model(self, batch, recoder=None, current_step=0, pbar=None):
        self.model.train()
        with autocast():
            loss, acc = self.model(batch)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        self.scheduler.step()
        if recoder:
            recoder.add_scalar(f'train/RunLoss', loss.item(), current_step)
            recoder.add_scalar(f'train/Acc', acc, current_step)
        pbar.set_description(f'[!] loss: {round(loss.item(), 4)}; acc: {round(acc*100, 2)}')
        pbar.update(1)

    def save_model(self, question_path, answer_path):
        self.model.module.question_encoder.save_pretrained(question_path)
        self.model.module.answer_encoder.save_pretrained(answer_path)
        print(f'[!] save model into:\n >> {question_path}\n >> {answer_path}')
 
    def set_optimizer_scheduler_ddp(self):
        if self.args['mode'] in ['train']:
            self.optimizer = transformers.AdamW(
                self.model.parameters(), 
                lr=self.args['lr'],
            )
            self.scaler = GradScaler()
            self.scheduler = transformers.get_linear_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=self.args['warmup_step'], 
                num_training_steps=self.args['total_step'],
            )
            self.model = nn.parallel.DistributedDataParallel(
                self.model, 
                device_ids=[self.args['local_rank']], 
                output_device=self.args['local_rank'],
                find_unused_parameters=True,
            )

