import argparse
import os

import torch
import wandb
from torch import nn
from torch.utils import data
from torch.utils.data import DistributedSampler
import torch.multiprocessing as mp

from Trainer import Trainer
from core.logger import InfoLogger, VisualWriter
import core.parser as Parser
import core.util as Util
from diffusion.gaussian_diffusion import GaussianDiffusion, get_beta_schedule
from core.parser import init_obj
from diffusion import gaussian_diffusion as gd
from mri_utils import ksp_to_viewable_image

def mae(input, target):
    with torch.no_grad():
        loss = nn.L1Loss()
        output = loss(input, target)
    return output


def define_network(logger, opt, network_opt):
    """ define network with weights initialization """
    net = init_obj(network_opt, logger)
    if opt['phase'] == 'train':
        logger.info('Network [{}] weights initialize using [{:s}] method.'.format(net.__class__.__name__,
                                                                                  network_opt['args'].get('init_type',
                                                                                                          'default')))
    return net


def main_worker(gpu, opt, wandb_run=''):
    if 'local_rank' not in opt:
        opt['local_rank'] = opt['global_rank'] = gpu
    if opt['distributed']:
        torch.cuda.set_device(int(opt['local_rank']))
        print('using GPU {} for training'.format(int(opt['local_rank'])))
        torch.distributed.init_process_group(backend='nccl',
                                             init_method=opt['init_method'],
                                             world_size=opt['world_size'],
                                             rank=opt['global_rank'],
                                             group_name='mtorch'
                                             )
    '''set seed and and cuDNN environment '''
    torch.backends.cudnn.enabled = True
    # warnings.warn('You have chosen to use cudnn for accleration. torch.backends.cudnn.enabled=True')
    Util.set_seed(opt['seed'])
    phase_logger = InfoLogger(opt)
    phase_writer = VisualWriter(opt, phase_logger)
    phase_logger.info('Create the log file in directory {}.\n'.format(opt['path']['experiments_root']))

    # Load model:
    model = define_network(phase_logger, opt, opt['model']['network'])

    mean_type = (opt['model']['mean_type'] if 'mean_type' in opt['model'] else "eps")
    mean_type = {"eps": gd.ModelMeanType.EPSILON, "x": gd.ModelMeanType.START_X}[mean_type]
    diffusion = GaussianDiffusion(betas=get_beta_schedule(**opt['model']['diffusion']['beta_schedule']),
                                  model_mean_type=mean_type,
                                  model_var_type=gd.ModelVarType.FIXED_LARGE,
                                  loss_type=gd.LossType.MSE)

    train_dataset = init_obj(opt['datasets']['train']['which_dataset'], phase_logger, default_file_name='data.dataset', init_type='Dataset')
    val_dataset = init_obj(opt['datasets']['validation']['which_dataset'], phase_logger, default_file_name='data.dataset', init_type='Dataset')

    data_sampler = None
    loader_opts = dict(**opt['datasets']['train']['dataloader']['args'])
    val_loader_opts = dict(**opt['datasets']['validation']['dataloader']['args'])
    if opt['distributed']:
        data_sampler = DistributedSampler(train_dataset,
                                          shuffle=opt['datasets']['train']['dataloader']['args']['shuffle'],
                                          num_replicas=opt['world_size'],
                                          rank=opt['global_rank'])
        loader_opts["shuffle"] = False

    train_loader = data.DataLoader(train_dataset, sampler=data_sampler, **loader_opts)
    val_loader = data.DataLoader(val_dataset, **val_loader_opts)

    base_change = opt['model']['base_change'] if 'base_change' in opt['model'] else None
    base_change = {None: None, "mri": ksp_to_viewable_image}[base_change]

    if gpu == 0 and wandb_run:
        wandb_run = wandb.init(project="GSURE-Diffusion", entity=wandb_run, config={})
        wandb_run.config.update(opt)
    else:
        wandb_run = None

    print("Dataset size: ", len(train_dataset))

    trainer = Trainer(
        network=model,
        diffusion=diffusion,
        phase_loader=train_loader,
        val_loader=val_loader,
        metrics=[mae],
        logger=phase_logger,
        writer=phase_writer,
        wandb_run=wandb_run,
        sample_num=opt['model']['diffusion']['beta_schedule']["num_diffusion_timesteps"],
        task="unconditional",
        optimizers=opt['model']['trainer']['args']['optimizers'],
        ema_scheduler=opt['model']['trainer']['args']['ema_scheduler'],
        sigma_0=opt['model']['trainer']['args']['sigma_0'],
        base_change=base_change,
        model_wrapper=(opt['model']['model_wrapper'] if 'model_wrapper' in opt['model'] else False),
        Lambda=(opt['model']['Lambda'] if 'Lambda' in opt['model'] else 1),
        gsure=(opt['model']['gsure'] if 'gsure' in opt['model'] else True),
        opt=opt
    )

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='JSON file for configuration')
    parser.add_argument('-b', '--batch', type=int, default=None, help='Batch size in every gpu')
    parser.add_argument('-p', '--phase', type=str, choices=['train'], help='Run train or test', default='train')
    parser.add_argument('-d', '--debug', action='store_true', help='Run script in debug setting')
    parser.add_argument('-P', '--port', default='21012', type=str, help='Port setting for DDP')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None, help='GPU numbers to use for training')
    parser.add_argument('--wandb', type=str, default='', help='W & B entity to use for wandb, leave empty for no W & B sync')

    ''' parser configs '''
    args = parser.parse_args()
    opt = Parser.parse(args)

    ''' cuda devices '''
    gpu_str = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
    print('export CUDA_VISIBLE_DEVICES={}'.format(gpu_str))

    ''' use DistributedDataParallel(DDP) and multiprocessing for multi-gpu training'''
    if opt['distributed']:
        ngpus_per_node = len(opt['gpu_ids'])
        opt['world_size'] = ngpus_per_node
        opt['init_method'] = 'tcp://127.0.0.1:' + args.port
        mp.spawn(main_worker, args=(ngpus_per_node, opt, args.wandb))
    else:
        opt['world_size'] = 1
        main_worker(0, opt, args.wandb)
