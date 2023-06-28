from header import *
from .dprdataset import *

def load_dataset(args):
    dataset_name = args['models'][args['model']]['dataset_name']
    dataset_t = globals()[dataset_name]
    path = f'{args["root_dir"]}/data/{args["dataset"]}/train.json'
    data = dataset_t(path, **args)
    if args['mode'] in ['train']:
        sampler = torch.utils.data.distributed.DistributedSampler(data)
        iter_ = DataLoader(data, batch_size=args['batch_size'], collate_fn=data.collate, sampler=sampler)
    else:
        iter_ = DataLoader(data, batch_size=args['batch_size'], collate_fn=data.collate)
        sampler = None
    return data, iter_, sampler
