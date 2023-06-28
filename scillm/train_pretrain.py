from header import *
from datasets import *
from model import *
from config import *


def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--model', type=str)
    parser.add_argument('--train_data_path', type=str)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--log_path', type=str)
    return parser.parse_args()

def initialize_distributed(args):
    args['master_ip'] = os.getenv('MASTER_ADDR', 'localhost')
    args['master_port'] = os.getenv('MASTER_PORT', '6000')
    args['world_size'] = int(os.getenv('WORLD_SIZE', '1'))
    args['local_rank'] = int(os.getenv('RANK', '0')) % torch.cuda.device_count()
    device = args['local_rank'] % torch.cuda.device_count()
    torch.cuda.set_device(device)
    deepspeed.init_distributed(dist_backend='nccl')

def set_random_seed(seed):
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

def config_env(args):
    # default root_dir is ..
    args['root_dir'] = '../'
    args['mode'] = 'train'
    config = load_config(args)
    args.update(config)
    initialize_distributed(args)
    set_random_seed(args['seed'])

def build_directory(path):
    if os.path.exists(path):
        pass
    else: # recursively construct directory
        os.makedirs(path, exist_ok=True)

def main(**args):
    config_env(args)
    args['ds_config_path'] = f'dsconfig/{args["model"]}.json'
    dschf = HfDeepSpeedConfig(args['ds_config_path'])
    args['dschf'] = dschf

    build_directory(args['save_path'])
    build_directory(args['log_path'])
    args['save_counter'] = 0

    # load train split
    args['data_path'] = os.path.join(args["train_data_path"], f'split_0{args["local_rank"]}')
    train_data, train_iter, sampler = load_dataset(args)

    length = args['total_step']
    args['total_steps'] = int(args['total_step']/8)
    agent = load_model(args)
    torch.distributed.barrier()

    # init the tensorboard
    if args['local_rank'] == 0:
        sum_writer = SummaryWriter(log_dir=args["log_path"])
    else:
        sum_writer = None

    # set the evaluation step
    args['eval_and_save_steps'] = set([int(length * i) for i in np.arange(0, 1, args['eval_interval'])][1:])
    args['eval_and_save_steps'].add(length)
    print(f'[!] evaluate step: {args["eval_and_save_steps"]}')

    # begin to train
    pbar = tqdm(total=length)    # maximum total number
    current_step = 0

    while True:
        for batch in train_iter:
            agent.train_model(
                batch,
                current_step=current_step,
                pbar=pbar,
                sum_writer=sum_writer
            )
            current_step += 1
        if current_step > length:
            break

if __name__ == "__main__":
    args = parser_args()
    args = vars(args)
    main(**args)
