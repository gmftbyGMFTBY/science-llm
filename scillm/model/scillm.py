from header import *
from .utils import *
# from .llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

class SciLLM(nn.Module):

    def __init__(self, **args):
        super(SciLLM, self).__init__()
        self.args = args

        # TODO: replace_llama_attn_with_flash_attn()
        # model loading
        self.model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=args['model_path'],
            load_in_4bit=True,
            max_memory={i: '24576MB' for i in range(torch.cuda.device_count())},
            torch_dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            )
        )

        # peft preparation
        self.model.gradient_checkpointing_enable()
        self.model = prepare_model_for_kbit_training(self.model)

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
            input_ids=inputs['input_ids'].to(f"cuda:{self.args['local_rank']}"),
            attention_mask=inputs['attention_mask'].to(f"cuda:{self.args['local_rank']}"),
            labels=inputs['labels'].to(f"cuda:{self.args['local_rank']}")
        )
        loss = outputs.loss
        
        # monitor token accuracy
        logits = outputs.logits[:, :-1, :]
        labels = inputs['labels'][:, 1:].to(f"cuda:{self.args['local_rank']}")
        token_acc = monitor_token_acc(logits, labels)
        return loss, token_acc
 
