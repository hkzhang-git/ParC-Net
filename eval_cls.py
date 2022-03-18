
import torch
import multiprocessing

from cvnets import get_model
from data import create_eval_loader
from engine import Evaluator

from options.opts import get_eval_arguments
from utils import logger
from utils.common_utils import device_setup, create_directories
from utils.ddp_utils import is_master, distributed_init


def main(opts, **kwargs):
    num_gpus = getattr(opts, "dev.num_gpus", 0)  # defaults are for CPU
    dev_id = getattr(opts, "dev.device_id", torch.device('cpu'))
    device = getattr(opts, "dev.device", torch.device('cpu'))
    is_distributed = getattr(opts, "ddp.use_distributed", False)

    # set-up data loaders
    val_loader = create_eval_loader(opts)

    # set-up the model
    model = get_model(opts)

    is_master_node = is_master(opts)
    if num_gpus <= 1:
        model = model.to(device=device)
    elif is_distributed:
        model = model.to(device=device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dev_id], output_device=dev_id)
        if is_master_node:
            logger.log('Using DistributedDataParallel for evaluation')
    else:
        model = torch.nn.DataParallel(model)
        model = model.to(device=device)
        if is_master_node:
            logger.log('Using DataParallel for evaluation')

    eval_engine = Evaluator(opts=opts, model=model, eval_loader=val_loader)
    eval_engine.run()

def main_worker_classification(**kwargs):
    opts = get_eval_arguments()
    print(opts)
    # device set-up
    opts = device_setup(opts)

    node_rank = getattr(opts, "ddp.rank", 0)
    if node_rank < 0:
        logger.error('--rank should be >=0. Got {}'.format(node_rank))

    is_master_node = is_master(opts)

    # create the directory for saving results
    save_dir = getattr(opts, "common.results_loc", "results")
    run_label = getattr(opts, "common.run_label", "run_1")
    exp_dir = '{}/{}'.format(save_dir, run_label)
    setattr(opts, "common.exp_loc", exp_dir)
    create_directories(dir_path=exp_dir, is_master_node=is_master_node)

    world_size = getattr(opts, "ddp.world_size", 1)
    num_gpus = 1


    # No of data workers = no of CPUs (if not specified or -1)
    n_cpus = multiprocessing.cpu_count()
    dataset_workers = getattr(opts, "dataset.workers", -1)


    if dataset_workers == -1:
        setattr(opts, "dataset.workers", n_cpus)

    # adjust the batch size
    train_bsize = getattr(opts, "dataset.train_batch_size0", 32) * max(1, num_gpus)
    val_bsize = getattr(opts, "dataset.val_batch_size0", 32) * max(1, num_gpus)
    setattr(opts, "dataset.train_batch_size0", train_bsize)
    setattr(opts, "dataset.val_batch_size0", val_bsize)
    setattr(opts, "dev.device_id", None)
    main(opts=opts, **kwargs)


if __name__ == "__main__":
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main_worker_classification()
