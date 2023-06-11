from header import *
from config import *
from model import *
from datasets import *

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--model', default='decapoda-research/llama-7b-hf', type=str)
    parser.add_argument('--model_path', default='decapoda-research/llama-7b-hf', type=str)
    parser.add_argument('--data_path', default='decapoda-research/llama-7b-hf', type=str)
    parser.add_argument('--delta_model_path', default='ckpt/scillm/pytorch_model.bin', type=str)
    return parser.parse_args()

def main_original(args):
    args['mode'] = 'test'
    args.update(load_config(args))
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args['model_path'],
        # load_in_4bit=True,
        # max_memory={i: '24576MB' for i in range(torch.cuda.device_count())},
        torch_dtype=torch.bfloat16,
        # quantization_config=BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type='nf4'
        # )
    ).cuda()
    tokenizer = LlamaTokenizer.from_pretrained(args['model_path'])
    ppl_criterion = nn.CrossEntropyLoss(reduction='none')
    print(f'[!] load model and tokenizer over')
    
    # load test dataset
    with torch.no_grad():
        _, test_iter, _ = load_dataset(args)
        losses = []
        for batch in tqdm(test_iter):
            outputs = model(
                input_ids=batch['input_ids'].cuda(),
                attention_mask=batch['attention_mask'].cuda(),
            )
            logits = outputs.logits[:, :-1, :]
            loss = ppl_criterion(logits.reshape(-1, logits.size(-1)), batch['labels'].cuda()[:, 1:].reshape(-1)).tolist()
            losses.extend(loss)
        ppl = np.exp(np.mean(losses))
        print(f'[!] ppl: {round(ppl, 4)}')



def main(args):
    args.update({
        'lora_r': 8,
        'lora_alpha': 32,
        'lora_dropout': 0.05,
        'mode': 'test'
    })
    args.update(load_config(args))
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args['model_path'],
        # load_in_4bit=True,
        # max_memory={i: '24576MB' for i in range(torch.cuda.device_count())},
        torch_dtype=torch.bfloat16,
        # quantization_config=BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type='nf4'
        # )
    ).cuda()

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=True,
        r=args['lora_r'],
        lora_alpha=args['lora_alpha'],
        lora_dropout=args['lora_dropout'],
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj']
    )

    model = PeftModel.from_pretrained(model, args['delta_model_path'])
    model = model.merge_and_unload()
    tokenizer = LlamaTokenizer.from_pretrained(args['model_path'])
    ppl_criterion = nn.CrossEntropyLoss(reduction='none')
    print(f'[!] load model and tokenizer over')
    
    # load test dataset
    with torch.no_grad():
        args['mode'] = 'test'
        _, test_iter, _ = load_dataset(args)
        losses = []
        for batch in tqdm(test_iter):
            outputs = model(
                input_ids=batch['input_ids'].cuda(),
                attention_mask=batch['attention_mask'].cuda(),
            )
            logits = outputs.logits[:, :-1, :]
            loss = ppl_criterion(logits.reshape(-1, logits.size(-1)), batch['labels'].cuda()[:, 1:].reshape(-1)).tolist()
            losses.extend(loss)
        ppl = np.exp(np.mean(losses))
        print(f'[!] ppl: {round(ppl, 4)}')

if __name__ == "__main__":
    args = vars(parser_args())
    main_original(args)
