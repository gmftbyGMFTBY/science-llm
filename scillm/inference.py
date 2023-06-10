from header import *
from model import *

args = {
    'lora_r': 8,
    'lora_dropout': 0.05,
    'lora_alpha': 32,
    'model_path': 'decapoda-research/llama-7b-hf'
}

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

peft_config = LoraConfig(
	task_type=TaskType.CAUSAL_LM,
	inference_mode=True,
	r=args['lora_r'],
	lora_alpha=args['lora_alpha'],
	lora_dropout=args['lora_dropout'],
	target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj']
)

model = PeftModel.from_pretrained(model, 'ckpt/scillm_backup/peft_model')
tokenizer = LlamaTokenizer.from_pretrained(args['model_path'])
print(f'[!] load model and tokenizer over')
inputs = tokenizer.encode('This paper propose', add_special_tokens=False)
generations = model.generate(input_ids=torch.LongTensor(inputs).unsqueeze(0).cuda(), max_new_tokens=128, use_cache=True)
output = tokenizer.decode(generations[0])
print(output)


def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--model_path', default='decapoda-research/llama-7b-hf', type=str)
    parser.add_argument('--delta_model_path', default='ckpt/scillm/pytorch_model.bin', type=str)
    parser.add_argument('--max_length', default=4096, type=int)
    parser.add_argument('--generate_len', default=512, type=int)
    parser.add_argument('--top_k', default=50, type=int)
    parser.add_argument('--top_p', default=0.92, type=float)
    return parser.parse_args()

def main(args):
    args.update({
        'lora_r': 8,
        'lora_alpha': 32,
        'lora_dropout': 0.05,
        'mode': 'inference'
    })
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

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=True,
        r=args['lora_r'],
        lora_alpha=args['lora_alpha'],
        lora_dropout=args['lora_dropout'],
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj']
    )

    model = PeftModel.from_pretrained(model, 'ckpt/scillm/0')
    tokenizer = LlamaTokenizer.from_pretrained(args['model_path'])
    print(f'[!] load model and tokenizer over')
    inputs = tokenizer.encode('This paper propose', add_special_tokens=False)

    while True:
        instruction = input('[!] Input Context: ')
        tokens = tokenizer.encode(instruction, add_special_tokens=False)
        tokens = torch.LongTensor(tokens[-args['max_length']+args['generate_len']:]).unsqueeze(0).cuda()

        length = len(tokens[0])
        with torch.no_grad():
            rest = model.model.generate(
                input_ids=tokens, 
                max_length=length+args['generate_len'], 
                use_cache=True, 
                do_sample=True, 
                top_p=args['top_p'], 
                top_k=args['top_k']
            )
        output = rest[0][length:]
        string = tokenizer.decode(output, skip_special_tokens=False)
        print(f'[!] Generation: {string}\n')

if __name__ == "__main__":
    args = vars(parser_args())
    main(args)
