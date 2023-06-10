from header import *
from .samplers import DistributedBatchSampler
from .sft_dataset import *
from .pretrain_dataset import *

def load_dataset(args):
    tokenizer = LlamaTokenizer.from_pretrained(args['model_path'])
    tokenizer.pad_token_id = tokenizer.eos_token_id
    if args['mode'] == 'test':
        dataset_name = args['models'][args['model']]['test_dataset']
    else:
        dataset_name = args['models'][args['model']]['dataset']
    args['tokenizer'] = tokenizer
    data = globals()[dataset_name](**args)

    sampler = torch.utils.data.DistributedSampler(data)
    # world_size = torch.distributed.get_world_size()
    # rank = torch.distributed.get_rank()
    # batch_size = args['world_size'] * args['dschf'].config['train_micro_batch_size_per_gpu']
    # batch_sampler = DistributedBatchSampler(
    #     sampler, 
    #     batch_size,
    #     True,
    #     rank,
    #     world_size
    # )
    iter_ = DataLoader(
        data, 
        batch_size=args['dschf'].config['train_micro_batch_size_per_gpu'],
        collate_fn=data.collate, 
        sampler=sampler
    )
    return data, iter_, sampler
