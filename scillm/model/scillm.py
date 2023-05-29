from header import *
from .utils import *

class SciLLM(nn.Module):

    def __init__(self, **args):
        super(SciLLM, self).__init__()
        self.args = args
        self.model = LlamaForCausalLM.from_pretrained(args['model_path'], torch_dtype=torch.float16)

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=self.args['lora_r'], 
            lora_alpha=self.args['lora_alpha'], 
            lora_dropout=self.args['lora_dropout'],
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
        )

        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        self.ppl_criterion = nn.CrossEntropyLoss(reduce='none')

    @torch.no_grad()
    def calculate_ppl(self, inputs):
        outputs = self.model(
            input_ids=inputs['input_ids'].cuda(),
            attention_mask=inputs['attention_mask'].cuda(),
            labels=inputs['labels'].cuda()
        )
        loss = outputs.loss.tolist()
        return loss

    def forward(self, inputs):
        outputs = self.model(
            input_ids=inputs['input_ids'].cuda(), 
            attention_mask=inputs['attention_mask'].cuda(), 
            labels=inputs['labels'].cuda()
        )
        loss = outputs.loss
        
        # monitor token accuracy
        logits = outputs.logits[:, :-1, :]
        labels = inputs['labels'][:, 1:]
        token_acc = monitor_token_acc(logits, labels)
        return loss, token_acc
 
