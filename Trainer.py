import torch
import tqdm
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import make_grid

from core.infra import Infra
from core.logger import LogTracker
import copy
import wandb

from mri_utils import FFT_Wrapper, FFT_NN_Wrapper


class EMA():
    def __init__(self, beta=0.9999):
        super().__init__()
        self.beta = beta
    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            if current_params.requires_grad:
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = self.update_average(old_weight, up_weight)
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Trainer(Infra):
    def __init__(self, network, diffusion, optimizers, sigma_0,
                 base_change, model_wrapper, Lambda, gsure, ema_scheduler=None, **kwargs):
        ''' must to init BaseModel with kwargs '''
        super(Trainer, self).__init__(**kwargs)

        ''' networks, dataloder, optimizers, losses, etc. '''
        self.netG = network
        self.diffusion = diffusion
        self.sigma_0 = sigma_0
        self.base_change = base_change
        self.Lambda = Lambda
        self.gsure = gsure
        if model_wrapper:
            print("Activating model FFT NN wrapper")
            self.netG = FFT_NN_Wrapper(self.netG)
        if ema_scheduler is not None:
            self.ema_scheduler = ema_scheduler
            self.netG_EMA = copy.deepcopy(self.netG)
            self.EMA = EMA(beta=self.ema_scheduler['ema_decay'])
        else:
            self.ema_scheduler = None
        
        ''' networks can be a list, and must convert by self.set_device function if using multiple GPU. '''
        self.netG = self.set_device(self.netG, distributed=self.opt['distributed'])
        if self.ema_scheduler is not None:
            self.netG_EMA = self.set_device(self.netG_EMA, distributed=self.opt['distributed'])

        self.load_networks()


        self.optG = torch.optim.Adam(self.netG.parameters(), **optimizers[0])
        self.optimizers.append(self.optG)
        # self.schedulers.append(LambdaLR(self.optG, lr_lambda=lambda epoch: max(epoch / 10, 0.1) if epoch <= 10 else 1))
        self.resume_training()

        ''' can rewrite in inherited class for more informations logging '''
        self.train_metrics = LogTracker(*["mse"], phase='train')
        self.val_metrics = LogTracker(*[m.__name__ for m in self.metrics], phase='val')

        
    def set_input(self, data):
        ''' must use set_device in tensor '''
        with torch.no_grad():
            if isinstance(data, dict):
                img = data["image"]
                mask = data["mask"]
            else:
                img = data
                mask = torch.ones_like(data)
            self.image = self.set_device(img)
            self.mask = self.set_device(mask.long())
            self.class_label = self.set_device(torch.tensor([0]))
            self.batch_size = img.shape[0]
    
    def get_current_visuals(self, phase='train'):
        dict = {
            'gt_image': (self.gt_image.detach()[:].float().cpu()+1)/2,
        }

        if phase != 'train':
            dict.update({
                'output': (self.output.detach()[:].float().cpu()+1)/2
            })
        return dict


    def train_step(self):
        self.netG.train()
        self.train_metrics.reset()
        self.optG.zero_grad()
        for train_data in tqdm.tqdm(self.phase_loader):
            self.optG.zero_grad()
            self.set_input(train_data)
            if self.gsure:
                loss = self.diffusion.training_losses_gsure(model=self.netG, x_start=self.image, mask=self.mask,
                                                  sigma_0=self.sigma_0, Lambda=self.Lambda)
            else:
                loss = self.diffusion.training_losses(model=self.netG, x_start=self.image)
            loss.backward()
            self.iter += 1
            self.optG.step()
            self.writer.set_iter(self.epoch, self.iter, phase='train')
            self.train_metrics.update("mse", loss.item())
            if self.iter % self.opt['train']['log_iter'] == 0:
                for key, value in self.train_metrics.result().items():
                    self.logger.info('{:5s}: {}\t'.format(str(key), value))
                    self.writer.add_scalar(key, value)
                if self.wandb_run is not None and self.wandb_run:
                    if self.schedulers:
                        base = {"batch": self.iter, "epoch": self.epoch, "learning-rate": self.schedulers[0].get_last_lr()[0]}
                    else:
                        base = {"batch": self.iter, "epoch": self.epoch}
                    base.update(self.train_metrics.result())
                    self.wandb_run.log(base)
                self.train_metrics.reset()
            if self.ema_scheduler is not None:
                if self.iter > self.ema_scheduler['ema_start'] and self.iter % self.ema_scheduler['ema_iter'] == 0:
                    self.EMA.update_model_average(self.netG_EMA, self.netG)

        for scheduler in self.schedulers:
            scheduler.step()
        return self.train_metrics.result()

    def val_step(self):
        self.netG.eval()
        self.val_metrics.reset()
        with torch.no_grad():
            for i, val_data in tqdm.tqdm(enumerate(self.val_loader)):
                self.set_input(val_data)
                clean_image, noisy_image = self.diffusion.clean_image(model=self.netG, x_start=self.image,
                                                  with_P=self.with_P)
                self.writer.set_iter(self.epoch, self.iter, phase='val')
                for met in self.metrics:
                    key = met.__name__
                    value = met(self.image, clean_image)
                    self.val_metrics.update(key, value)
                    self.writer.add_scalar(key, value)
                if self.wandb_run is not None and self.wandb_run:
                    if self.base_change is not None:
                        self.image, noisy_image, clean_image = self.base_change(self.image), self.base_change(noisy_image), self.base_change(clean_image)
                    base = {"epoch": self.epoch}
                    base.update(self.val_metrics.result())
                    im = make_grid((0.5 + 0.5 * torch.cat([self.image, noisy_image, clean_image], dim=0)).clamp(0, 1), nrow=self.batch_size, padding=0)
                    images = wandb.Image(im, caption="Top: Output, Bottom: Input")
                    base.update({"images": images})
                    self.wandb_run.log(base)
        return self.val_metrics.result()

    def load_networks(self):
        """ save pretrained model and training state, which only do on GPU 0. """
        if self.opt['distributed']:
            netG_label = self.netG.module.__class__.__name__
        else:
            netG_label = self.netG.__class__.__name__
        self.load_network(network=self.netG, network_label=netG_label, strict=True)
        if self.ema_scheduler is not None:
            self.load_network(network=self.netG_EMA, network_label=netG_label+'_ema', strict=True)

    def save_everything(self):
        """ load pretrained model and training state. """
        if self.opt['distributed']:
            netG_label = self.netG.module.__class__.__name__
        else:
            netG_label = self.netG.__class__.__name__
        self.save_network(network=self.netG, network_label=netG_label)
        if self.ema_scheduler is not None:
            self.save_network(network=self.netG_EMA, network_label=netG_label+'_ema')
        self.save_training_state()
