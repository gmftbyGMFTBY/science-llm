from header import *
from config import *
from model import *
from datasets import *

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = stops
        self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        stop_count = 0
        for stop in self.stops:
            stop_count = (stop == input_ids[0]).sum().item()
        if stop_count >= self.ENCOUNTERS:
            return True
        return False

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--model', default='decapoda-research/llama-7b-hf', type=str)
    parser.add_argument('--model_path', default='decapoda-research/llama-7b-hf', type=str)
    parser.add_argument('--data_path', default='decapoda-research/llama-7b-hf', type=str)
    parser.add_argument('--delta_model_path', default='ckpt/scillm/pytorch_model.bin', type=str)
    parser.add_argument('--result_path', default='decapoda-research/llama-7b-hf', type=str)
    parser.add_argument('--recall', default='False', type=str)
    return parser.parse_args()


def main_emotional(args):
    args.update({
        'lora_r': 72,
        'lora_alpha': 16,
        'lora_dropout': 0.1,
        'mode': 'test'
    })
    args.update(load_config(args))
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

    model = prepare_model_for_kbit_training(model)
    model = PeftModel.from_pretrained(model, args['delta_model_path'])
    ppl_criterion = nn.CrossEntropyLoss(reduction='none')
    print(f'[!] load model and tokenizer over')

    with torch.no_grad():
        results = []
        # stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=[0], encounters=1)])
        # stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=[0], encounters=3)])
        history = []
        while True:
            utterance = input('Human: ')
            if 'exit' in utterance.lower():
                print(f'[!] end the session')
                break
            utterance = 'Human: ' + utterance
            history.append(utterance)
            string = tokenizer.eos_token.join(history) + f'{tokenizer.eos_token}Bot:'
            tokens = tokenizer.encode(string, add_special_tokens=False)[-args['max_seq_length']:]
            tokens = torch.LongTensor(tokens).cuda()
            length = len(tokens)
            tokens = tokens.unsqueeze(0)

            # greedy search
            outputs = model.generate(
                input_ids=tokens,
                max_new_tokens=128,
                return_dict_in_generate=True,
                stopping_criteria=stopping_criteria
            )
            generation = tokenizer.decode(outputs['sequences'][0, length:], skip_special_tokens=True)
            history.append(f'Bot: {generation}')
            print(f'Bot: {generation}')



def main(args):
    args.update({
        'lora_r': 64,
        'lora_alpha': 16,
        'lora_dropout': 0.1,
        'mode': 'test',
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

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=True,
            r=args['lora_r'],
            lora_alpha=args['lora_alpha'],
            lora_dropout=args['lora_dropout'],
            target_modules=['o_proj', 'W_pack', 'gate_proj', 'down_proj', 'up_proj']
        )

    model = prepare_model_for_kbit_training(model)
    model = PeftModel.from_pretrained(model, args['delta_model_path'])
    ppl_criterion = nn.CrossEntropyLoss(reduction='none')
    print(f'[!] load model and tokenizer over')

    # load test dataset
    with open(args['data_path']) as f:
        question_set = set()
        data = []
        for sample in json.load(f):
            if sample['question'] not in question_set:
                question_set.add(sample['question'])
                data.append(sample)

    with torch.no_grad():
        results = []
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=[2], encounters=1)])
        f = open(args['result_path'], 'w')
        for item in tqdm(data):
            q = item['question']
            if args['recall'] == 'True':
                evidence = '\n'.join(item['recall_evidence'])
            elif args['recall'] == 'False':
                evidence = item['evidence']
            else:
                raise Exception(f'[!] Unknown recall mode: {args["recall"]}')
            prompt = f'### Evidence:\n{evidence}\n\n### Instruction:\n{q}\n\n### Response:\n' 
            tokens = tokenizer.encode(prompt, add_special_tokens=False)[-args['max_seq_length']:]
            tokens = torch.LongTensor(tokens).cuda()
            length = len(tokens)
            tokens = tokens.unsqueeze(0)

            # greedy search
            outputs = model.generate(
                input_ids=tokens,
                max_new_tokens=128,
                return_dict_in_generate=True,
                stopping_criteria=stopping_criteria,
            )
            ipdb.set_trace()
            generation = tokenizer.decode(outputs['sequences'][0, length:], skip_special_tokens=True)
            result_item = deepcopy(item)
            result_item['generation'] = generation
            f.write(json.dumps(result_item) + '\n')
            f.flush()

if __name__ == "__main__":
    args = vars(parser_args())
    main(args)
    # main_emotional(args)
