from header import *

class SciDPR(nn.Module):

    def __init__(self, **args):
        super(SciDPR, self).__init__()
        self.args = args
        self.question_encoder = DPRQuestionEncoder.from_pretrained(self.args['question_model'])
        self.answer_encoder = DPRContextEncoder.from_pretrained(self.args['answer_model']) 
        total = sum([param.nelement() for param in self.parameters()])
        print('[!] Model Size: %2fM' % (total/1e6))

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.question_encoder(input_ids=cid, attention_mask=cid_mask).pooler_output
        rid_rep = self.answer_encoder(input_ids=rid, attention_mask=rid_mask).pooler_output
        return cid_rep, rid_rep

    @torch.no_grad()
    def get_answer_embedding(self, ids, attn_mask):
        rid_rep = self.answer_encoder(input_ids=ids, attention_mask=attn_mask).pooler_output
        return rid_rep

    @torch.no_grad()
    def get_question_embedding(self, ids, attn_mask):
        cid_rep = self.question_encoder(input_ids=ids, attention_mask=attn_mask).pooler_output
        return cid_rep

    def forward(self, batch):
        cid = batch['q_ids']
        rid = batch['a_ids']
        cid_mask = batch['q_mask']
        rid_mask = batch['a_mask']
        labels = batch['labels']

        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)

        dot_product = torch.matmul(cid_rep, rid_rep.t()) 
        batch_size = len(cid_rep)
        assert batch_size == len(labels)

        # constrastive loss
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), labels] = 1. 
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()

        # acc
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == labels).sum().item()
        acc = acc_num / batch_size

        return loss, acc
