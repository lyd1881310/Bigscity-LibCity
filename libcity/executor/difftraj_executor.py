import numpy as np
import os
import copy
from logging import getLogger
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
import torch.nn.functional as F
from tqdm import tqdm

from libcity.executor.abstract_executor import AbstractExecutor
from libcity.utils import get_evaluator


class DiffTrajExecutor(AbstractExecutor):
    def __init__(self, config, model, data_feature):
        self.config = config
        self.device = config['device']
        self.n_steps = config['num_diffusion_timesteps']
        self.model = model.to(self.device)
        self.beta = torch.linspace(config['beta_start'], config['beta_end'], self.n_steps).to(self.device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=self.config['lr'])
        self.use_ema = config['ema']
        if self.use_ema:
            self.ema_helper = EMAHelper(mu=config['ema_rate'])
            self.ema_helper.register(self.model)
        else:
            self.ema_helper = None
        self.evaluator = get_evaluator(config)
        self.exp_id = self.config['exp_id']
        self.cache_dir = './libcity/cache/{}/model_cache'.format(self.exp_id)
        self.evaluate_res_dir = './libcity/cache/{}/evaluate_cache'.format(self.exp_id)
        self._logger = getLogger()

    def train(self, train_dataloader, eval_dataloader):
        """
        use data to train model with config

        Args:
            train_dataloader(torch.Dataloader): Dataloader
            eval_dataloader(torch.Dataloader): Dataloader
        """
        for epoch in range(self.config['n_epochs']):
            loss = self._train_epoch(train_dataloader)
            self._logger.info(f'Epoch {epoch} loss {loss: .6f}')

    def _train_epoch(self, train_loader):
        avg_loss = 0
        for index, batch in enumerate(train_loader):
            batch = batch['gps']
            t = torch.randint(low=0, high=self.n_steps, size=(len(batch) // 2 + 1, )).to(self.device)
            t = torch.cat([t, self.n_steps - t - 1], dim=0)[:len(batch)]
            # Get the noised images (xt) and the noise (our target)
            xt, noise = self._q_xt_x0(batch, t)
            # Run xt through the network to get its predictions
            pred_noise = self.model(xt.float(), t)
            # Compare the predictions with the targets
            loss = F.mse_loss(noise.float(), pred_noise)
            # self._logger.info("train batch index: %d, MSE Loss: %.8f" % (index, loss.item()))
            # Store the loss for later viewing
            avg_loss += loss.item()
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            if self.use_ema:
                self.ema_helper.update(self.model)
        return avg_loss

    def _q_xt_x0(self, x0, t):
        # Modified to return the noise itself as well
        mean = self._gather(self.alpha_bar, t) ** 0.5 * x0
        var = 1 - self._gather(self.alpha_bar, t)
        eps = torch.randn_like(x0).to(self.device)
        return mean + (var ** 0.5) * eps, eps  # also returns noise

    @staticmethod
    def _gather(consts: torch.Tensor, t: torch.Tensor):
        c = consts.gather(-1, t)
        return c.reshape(-1, 1, 1)

    def evaluate(self, test_dataloader):
        """
        use model to test data
        Args:
            test_dataloader(torch.Dataloader): Dataloader
        """
        for batch in tqdm(test_dataloader, total=len(test_dataloader), desc='DiffTraj generating'):
            gen_traj = self.model.generate(batch)
            self.evaluator.collect({'gen': gen_traj, 'real': batch['loc_list']})
        self.evaluator.evaluate()
        self.evaluator.save_result(self.evaluate_res_dir, 'difftraj_evaluate')

    def load_model(self, cache_name):
        """
        加载对应模型的 cache
        Args:
            cache_name(str): 保存的文件名
        """
        state = torch.load(cache_name, map_location=self.device)
        self.model.load_state_dict(state)

    def save_model(self, cache_name):
        """
        将当前的模型保存到文件
        Args:
            cache_name(str): 保存的文件名
        """
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        torch.save(self.model.state_dict(), cache_name)


class EMAHelper(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (
                    1. -
                    self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(
                inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        # module_copy = copy.deepcopy(module)
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict
