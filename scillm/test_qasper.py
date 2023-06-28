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
    if args['base_model_name'] == 'llama':
        model = LlamaForCausalLM.from_pretrained(
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
        tokenizer = LlamaTokenizer.from_pretrained(args['model_path'])
        indexes = [3782, 8241]
    else:
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=args['model_path'],
            load_in_4bit=True,
            max_memory={i: '24576MB' for i in range(torch.cuda.device_count())},
            torch_dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            ),
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(args['model_path'], trust_remote_code=True)
        indexes = [4094, 9446]

    ppl_criterion = nn.CrossEntropyLoss(reduction='none')
    print(f'[!] load model and tokenizer over')

    # load test dataset
    with torch.no_grad():
        args['mode'] = 'test'
        _, test_iter, _ = load_dataset(args)
        acc = []
        pbar = tqdm(test_iter)
        for batch in pbar:
            length = len(batch['input_ids'][0])
            outputs = model.generate(
                input_ids=batch['input_ids'].cuda(),
                max_new_tokens=1,
                output_scores=True,
                return_dict_in_generate=True
            )
            result = tokenizer.decode(outputs['sequences'][0])
            logits = outputs.scores[-1]
            logits = logits[0, indexes]
            max_ = logits.argmax(dim=-1).item()
            if max_ == batch['labels'][0]:
                acc.append(1)
            else:
                acc.append(0)
            pbar.set_description(f'{round(np.mean(acc), 4)}')
    print(f'[!] accuracy is: {round(np.mean(acc), 4)}')
    

def main(args):
    args.update({
        'lora_r': 64,
        'lora_alpha': 16,
        'lora_dropout': 0.1,
        'mode': 'test'
    })
    args.update(load_config(args))
    if args['base_model_name'] == 'llama':
        model = LlamaForCausalLM.from_pretrained(
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
        tokenizer = LlamaTokenizer.from_pretrained(args['model_path'])

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=True,
            r=args['lora_r'],
            lora_alpha=args['lora_alpha'],
            lora_dropout=args['lora_dropout'],
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj']
        )
        indexes = [3782, 8241]
    else:
        args['lora_r'] = 72
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=args['model_path'],
            load_in_4bit=True,
            max_memory={i: '24576MB' for i in range(torch.cuda.device_count())},
            torch_dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            ),
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(args['model_path'], trust_remote_code=True)

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=True,
            r=args['lora_r'],
            lora_alpha=args['lora_alpha'],
            lora_dropout=args['lora_dropout'],
            target_modules=['o_proj', 'W_pack', 'gate_proj', 'down_proj', 'up_proj']
        )
        indexes = [4094, 9446]

    model = prepare_model_for_kbit_training(model)
    model = PeftModel.from_pretrained(model, args['delta_model_path'])
    ppl_criterion = nn.CrossEntropyLoss(reduction='none')
    print(f'[!] load model and tokenizer over')
    
    # load test dataset
    with torch.no_grad():
        args['mode'] = 'test'
        _, test_iter, _ = load_dataset(args)
        acc = []
        pbar = tqdm(test_iter)
        for batch in pbar:
            length = len(batch['input_ids'][0])
            outputs = model.generate(
                input_ids=batch['input_ids'].cuda(),
                max_new_tokens=1,
                output_scores=True,
                return_dict_in_generate=True
            )
            result = tokenizer.decode(outputs['sequences'][0])
            logits = outputs.scores[-1]
            logits = logits[0, indexes]
            max_ = logits.argmax(dim=-1).item()
            if max_ == batch['labels'][0]:
                acc.append(1)
            else:
                acc.append(0)
            pbar.set_description(f'{round(np.mean(acc), 4)}')
    print(f'[!] accuracy is: {round(np.mean(acc), 4)}')

if __name__ == "__main__":
    args = vars(parser_args())
    # main_original(args)
    main(args)
