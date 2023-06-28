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
    parser.add_argument('--port', default=23330, type=int)
    parser.add_argument('--model', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--delta_model_path', type=str)
    return parser.parse_args()


def create_emotional_app(args):
    app = Flask(__name__)
    # create the model
    args.update({
        'lora_r': 72,
        'lora_alpha': 16,
        'lora_dropout': 0.1,
        'mode': 'test'
    })
    args.update(load_config(args))
    if args['base_model_name'] == 'llama':
        tokenizer = LlamaTokenizer.from_pretrained(args['model_path'])
        tokenizer.add_special_tokens({"eos_token": "</s>"})
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
    else:
        tokenizer = AutoTokenizer.from_pretrained(args['model_path'], trust_remote_code=True)
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

    @app.route('/paper_ground_dialog', methods=['POST'])
    def generation_paper_ground_dialog_api():
        bt = time.time()
        with torch.no_grad():
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=[2], encounters=1)])
            data = json.loads(request.data)
            question = data['question']
            evidence = '\n'.join(data['evidences'])
            prompt_input = '### Evidence:\n{evidence}\n\n### Instruction:\n{question}\n\n### Response:\n'
            string = prompt_input.format_map({'question': question, 'evidence': evidence})
            tokens = tokenizer.encode(string, add_special_tokens=False)[-args['max_seq_length']:]
            tokens = torch.LongTensor(tokens).cuda()
            length = len(tokens)
            tokens = tokens.unsqueeze(0)

            # greedy search
            is_succ = True
            if data['decoding_method'] == 'greedy':
                outputs = model.generate(
                    input_ids=tokens,
                    max_new_tokens=data['max_new_tokens'],
                    return_dict_in_generate=True,
                    stopping_criteria=stopping_criteria
                )
            elif data['decoding_method'] == 'sampling':
                outputs = model.generate(
                    input_ids=tokens,
                    max_new_tokens=data['max_new_tokens'],
                    return_dict_in_generate=True,
                    stopping_criteria=stopping_criteria,
                    do_sample=True,
                    top_k=data['top_k'],
                    top_p=data['top_p']
                )
            elif data['decoding_method'] == 'contrastive':
                outputs = model.generate(
                    input_ids=tokens,
                    max_new_tokens=data['max_new_tokens'],
                    return_dict_in_generate=True,
                    stopping_criteria=stopping_criteria,
                    penalty_alpha=data['penalty_alpha'],
                    top_k=data['top_k']
                )
            else:
                print(f'[!] Unknown decoding method: {args["decoding_method"]}')
                is_succ = False
            if is_succ:
                generation = tokenizer.decode(outputs['sequences'][0, length:], skip_special_tokens=True)
            else:
                generation = ''
            result = {
                'time': time.time() - bt,
                'is_succ': is_succ,
                'generation': generation
            }
            return jsonify(result)

    @app.route('/emotional_dialog', methods=['POST'])
    def generation_dialog_api():
        bt = time.time()
        with torch.no_grad():
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=[0], encounters=1)])
            data = json.loads(request.data)
            history = data['history']
            string = tokenizer.eos_token.join(history) + f'{tokenizer.eos_token}Bot:'
            tokens = tokenizer.encode(string, add_special_tokens=False)[-args['max_seq_length']:]
            tokens = torch.LongTensor(tokens).cuda()
            length = len(tokens)
            tokens = tokens.unsqueeze(0)

            # greedy search
            is_succ = True
            if data['decoding_method'] == 'greedy':
                outputs = model.generate(
                    input_ids=tokens,
                    max_new_tokens=data['max_new_tokens'],
                    return_dict_in_generate=True,
                    stopping_criteria=stopping_criteria
                )
            elif data['decoding_method'] == 'sampling':
                outputs = model.generate(
                    input_ids=tokens,
                    max_new_tokens=data['max_new_tokens'],
                    return_dict_in_generate=True,
                    stopping_criteria=stopping_criteria,
                    do_sample=True,
                    top_k=data['top_k'],
                    top_p=data['top_p']
                )
            elif data['decoding_method'] == 'contrastive':
                outputs = model.generate(
                    input_ids=tokens,
                    max_new_tokens=data['max_new_tokens'],
                    return_dict_in_generate=True,
                    stopping_criteria=stopping_criteria,
                    penalty_alpha=data['penalty_alpha'],
                    top_k=data['top_k']
                )
            else:
                print(f'[!] Unknown decoding method: {args["decoding_method"]}')
                is_succ = False
            if is_succ:
                generation = tokenizer.decode(outputs['sequences'][0, length:], skip_special_tokens=True)
            else:
                generation = ''
            print('\n'.join(history) + '\n' + f'Bot: {generation}')
            result = {
                'time': time.time() - bt,
                'is_succ': is_succ,
                'generation': generation
            }
            return jsonify(result)

    return app


if __name__ == "__main__":
    args = vars(parser_args())
    app = create_emotional_app(args)
    app.run(
        host='0.0.0.0',
        port=args['port']
    )
