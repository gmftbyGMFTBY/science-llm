from header import *
from model import *

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
    tokenizer = LlamaTokenizer.from_pretrained(args['model_path'])
    model = SciLLM(**args).cuda().eval()
    model.load_state_dict(torch.load(f'{args["delta_model_path"]}/adapter_model.bin'), strict=False)
    print(f'[!] load model over')

    # instruction = 'What is the Natural Language Processing'
    while True:
        instruction = input('[!] Human: ')
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
        print(f'[!] Generation: {string}')

if __name__ == "__main__":
    args = vars(parser_args())
    main(args)
