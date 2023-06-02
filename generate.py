import argparse
import os
import shutil
from functools import partial

import torch

from mri_utils import ksp_to_viewable_image, FFT_Wrapper, FFT_NN_Wrapper
from torch_dct import idct_2d_shift

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch import nn
from torch.utils import data
from torch.utils.data import DistributedSampler, TensorDataset
import torch.nn.functional as F
import torch.multiprocessing as mp
from torchvision.utils import save_image
from tqdm import tqdm

from core.logger import InfoLogger, VisualWriter
import core.parser as Parser
import core.util as Util
from diffusion.gaussian_diffusion import GaussianDiffusion, get_beta_schedule
from core.parser import init_obj
from diffusion import gaussian_diffusion as gd, create_diffusion, create_diffusion


def mse_loss(output, target):
    return F.mse_loss(output, target)


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
        # net.init_weights()
    return net


def main_worker(gpu, opt, args):
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
    set_device = partial(Util.set_device, rank=opt["global_rank"])
    phase_logger = InfoLogger(opt)
    phase_logger.info('Create the log file in directory {}.\n'.format(opt['path']['experiments_root']))

    # Load model:
    model = define_network(phase_logger, opt, opt['model']['network'])
    state_dict = torch.load(args.model_path)
    if 'module.temb.dense.0.weight' in state_dict:
        # model saved as DDP
        state_dict = {k.replace("module.", ""): v for k, v  in state_dict.items()}
    if 'model.temb.dense.0.weight' in state_dict:
        # wrapper saved instead of model
        state_dict = {k.replace("model.", ""): v for k, v  in state_dict.items()}
    print(model.load_state_dict(state_dict))
    if (opt['model']['model_wrapper'] if 'model_wrapper' in opt['model'] else False):
        model = FFT_NN_Wrapper(model)
    model = set_device(model, distributed=opt['distributed'])
    model.eval()

    mean_type = (opt['model']['mean_type'] if 'mean_type' in opt['model'] else "eps")
    mean_type = {"eps": gd.ModelMeanType.EPSILON, "x": gd.ModelMeanType.START_X}[mean_type]
    diffusion = create_diffusion(str(args.steps), beta_sched_params=opt['model']['diffusion']['beta_schedule'],
                                 mean_type=mean_type)

    base_change = opt['model']['base_change'] if 'base_change' in opt['model'] else None
    base_change = {None: None, "mri": ksp_to_viewable_image}[base_change]

    if opt['global_rank'] == 0:
        if os.path.exists(args.output):
            shutil.rmtree(args.output)
        os.makedirs(args.output, exist_ok=True)
        if opt['distributed']:
            torch.distributed.barrier()
    else:
        if opt['distributed']:
            torch.distributed.barrier()
    dataset = TensorDataset(torch.arange(args.number))
    data_sampler = None
    if opt['distributed']:
        data_sampler = DistributedSampler(dataset, shuffle=False, num_replicas=opt['world_size'], rank=opt['global_rank'])
    class_loader = data.DataLoader(
        dataset,
        sampler=data_sampler,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )
    for (n,) in tqdm(class_loader):
        b = n.shape[0]
        z = torch.randn(b, opt['model']['network']['args']['out_channels'], *opt['datasets']['train']['which_dataset']['args']['image_size'])
        z = Util.set_device(z, distributed=opt['distributed'])
        if args.ddim:
            samples = diffusion.ddim_sample_loop(model, z.shape, z, clip_denoised=False,
                                                 progress=True, eta=args.eta, device=z.device)
        else:
            samples = diffusion.p_sample_loop(model, z.shape, z, clip_denoised=False,
                                              progress=True, device=z.device)
        if base_change is not None:
            samples = base_change(samples)
        for j in range(b):
            number = n[j].item()
            save_image((0.5 + 0.5 * samples[j]), os.path.join(args.output, f"{number:06d}.png"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='JSON file for configuration')
    parser.add_argument('-n', '--number', type=int, default=10000, help="Number of samples to generate")
    parser.add_argument('-m', '--model-path', type=str, required=True, help="Saved model checkpoint to use")
    parser.add_argument('-b', '--batch', type=int, default=64, help="Batch size")
    parser.add_argument('-o', '--output', type=str, required=True, help="Output path for generated samples")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('-s', '--steps', type=int, default=50, help="Number of steps")
    parser.add_argument('-e', '--eta', type=float, default=0.0, help="DDIM eta for DDIM sampling")
    parser.add_argument('--ddim', action='store_true', default=False, help="Use DDIM samples, do not use for DDPm sampling")
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('-P', '--port', default='21012', type=str)
    parser.add_argument('-gpu', '--gpu_ids', type=str, default="0", help="Numbers of GPUS to use")
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-p', '--phase', type=str, choices=['test'], help='Run train or test', default='test')


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
        mp.spawn(main_worker, args=(ngpus_per_node, opt, args))
    else:
        opt['world_size'] = 1
        main_worker(0, opt, args)
