import os
import sys
import json
import argparse
from easydict import EasyDict as edict

import torch
import numpy as np
import random

from trellis import models, datasets, trainers
from trellis.utils.dist_utils import setup_dist


def setup_rng(rank):
    torch.manual_seed(rank)
    torch.cuda.manual_seed_all(rank)
    np.random.seed(rank)
    random.seed(rank)


def get_model_summary(model):
    model_summary = 'Parameters:\n'
    model_summary += '=' * 128 + '\n'
    model_summary += f'{"Name":<{72}}{"Shape":<{32}}{"Type":<{16}}{"Grad"}\n'
    num_params = 0
    num_trainable_params = 0
    for name, param in model.named_parameters():
        model_summary += f'{name:<{72}}{str(param.shape):<{32}}{str(param.dtype):<{16}}{param.requires_grad}\n'
        num_params += param.numel()
        if param.requires_grad:
            num_trainable_params += param.numel()
    model_summary += '\n'
    model_summary += f'Number of parameters: {num_params}\n'
    model_summary += f'Number of trainable parameters: {num_trainable_params}\n'
    return model_summary


def main(local_rank, cfg):
    rank = cfg.node_rank * cfg.num_gpus + local_rank
    world_size = cfg.num_nodes * cfg.num_gpus
    if world_size > 1:
        setup_dist(rank, local_rank, world_size, cfg.master_addr, cfg.master_port)

    setup_rng(rank)

    dataset = getattr(datasets, cfg.dataset.name)(cfg.data_dir, **cfg.dataset.args)
    val_dataset_cfg = cfg.get('val_dataset', cfg.dataset)
    val_dataset = getattr(datasets, val_dataset_cfg.name)(cfg.val_data_dir, **val_dataset_cfg.args)

    model_dict = {
        name: getattr(models, model.name)(**model.args).cuda()
        for name, model in cfg.models.items()
    }

    if rank == 0:
        os.makedirs(cfg.output_dir, exist_ok=True)
        for name, backbone in model_dict.items():
            model_summary = get_model_summary(backbone)
            print(f'\n\nBackbone: {name}\n' + model_summary)
            with open(os.path.join(cfg.output_dir, f'{name}_model_summary.txt'), 'w') as fp:
                print(model_summary, file=fp)

    trainer = getattr(trainers, cfg.trainer.name)(
        model_dict,
        dataset,
        **cfg.trainer.args,
        val_dataset=val_dataset,
        output_dir=cfg.output_dir,
        load_dir='',
        step=None,
    )

    metrics = trainer.validate()
    if rank == 0:
        serializable = {
            key: (value.item() if isinstance(value, torch.Tensor) else float(value) if isinstance(value, np.generic) else value)
            for key, value in metrics.items()
        }
        with open(os.path.join(cfg.output_dir, 'metrics.json'), 'w') as fp:
            json.dump(serializable, fp, indent=2)
        print('\nSaved metrics to', os.path.join(cfg.output_dir, 'metrics.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--val_data_dir', type=str, default=None)
    parser.add_argument('--encoder_ckpt', type=str, required=True)
    parser.add_argument('--decoder_stage1_ckpt', type=str, required=True)
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--node_rank', type=int, default=0)
    parser.add_argument('--num_gpus', type=int, default=-1)
    parser.add_argument('--master_addr', type=str, default='localhost')
    parser.add_argument('--master_port', type=str, default='12345')
    opt = parser.parse_args()
    opt.num_gpus = torch.cuda.device_count() if opt.num_gpus == -1 else opt.num_gpus
    opt.val_data_dir = opt.val_data_dir or opt.data_dir

    config = json.load(open(opt.config, 'r'))
    cfg = edict()
    cfg.update(opt.__dict__)
    cfg.update(config)
    cfg.val_data_dir = opt.val_data_dir
    cfg.output_dir = opt.output_dir
    cfg.data_dir = opt.data_dir
    cfg.trainer.args['finetune_ckpt'] = {
        'encoder': opt.encoder_ckpt,
        'decoder_stage1': opt.decoder_stage1_ckpt,
    }
    cfg.trainer.args['max_steps'] = 0
    cfg.trainer.args['num_workers'] = 0
    cfg.trainer.args['persistent_workers'] = False
    cfg.trainer.args['batch_size_per_gpu'] = 1

    print('\n\nConfig:')
    print('=' * 80)
    print(json.dumps(cfg.__dict__, indent=4))

    if cfg.num_gpus > 1:
        torch.multiprocessing.spawn(main, args=(cfg,), nprocs=cfg.num_gpus, join=True)
    else:
        main(0, cfg)