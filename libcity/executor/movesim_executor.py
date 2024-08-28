import numpy as np
import os
import copy
from logging import getLogger

import torch
from torch.optim import Adam

from libcity.executor.abstract_executor import AbstractExecutor
from libcity.utils import get_evaluator


class MoveSimExecutor(AbstractExecutor):
    def __init__(self, config, model, data_feature):
        self.evaluator = get_evaluator(config)
        self.config = config
        self.data_feature = data_feature
        self.tim_num = config['tim_num']
        self.device = self.config['device']
        self.model = model.to(self.device)
        self.tmp_path = './libcity/tmp/checkpoint/'
        self.exp_id = self.config.get('exp_id', None)
        self.cache_dir = './libcity/cache/{}/model_cache'.format(self.exp_id)
        self.evaluate_res_dir = './libcity/cache/{}/evaluate_cache'.format(self.exp_id)
        self._logger = getLogger()

        self.optimizers = self._build_optimizer()

    def train(self, train_dataloader, eval_dataloader):
        """
        use data to train model with config

        Args:
            train_dataloader(torch.Dataloader): Dataloader
            eval_dataloader(torch.Dataloader): Dataloader
        """
        # self._pretrain_discriminator(train_dataloader)
        self._pretrain_generator(train_dataloader)
        self._reinforce_train(train_dataloader)

        # self.save_model()

    def _pretrain_discriminator(self, train_dataloader):
        for epoch in range(self.config['dis_pre_epoch']):
            avg_loss = self._train_dis_epoch(train_dataloader)
            self._logger.info(f'pretrain discriminator epoch {epoch} loss {avg_loss: .4f}')

    def _train_dis_epoch(self, train_dataloader):
        avg_loss = 0
        for batch in train_dataloader:
            batch_size, seq_len = batch['loc'].shape[0], batch['loc'].shape[1]
            gen_loc = self.model.generator.sample(batch_size, seq_len)
            dis_loss = self.model.calc_dis_loss({'real': batch['loc'], 'fake': gen_loc})
            self.optimizers['dis_pre'].zero_grad()
            dis_loss.backward()
            self.optimizers['dis_pre'].step()
            avg_loss += dis_loss.item()
        avg_loss /= len(train_dataloader)
        return avg_loss

    def _pretrain_generator(self, train_dataloader):
        """
        下一跳预测预训练
        """
        for epoch in range(self.config['gen_pre_epoch']):
            avg_loss = 0
            for itr, batch in enumerate(train_dataloader):
                # print(batch)
                gen_loss = self.model.calc_gen_pre_loss(batch)
                self.optimizers['gen_pre'].zero_grad()
                gen_loss.backward()
                self.optimizers['gen_pre'].step()
                avg_loss += gen_loss.item()
                # self._logger.info(f'loss {gen_loss.item(): .4f}')
                # if itr >= 10:
                #     return None
            avg_loss /= len(train_dataloader)
            self._logger.info(f'pretrain generator epoch {epoch} loss {avg_loss: .4f}')

    def _reinforce_train(self, train_dataloader):
        rollout = Rollout(model=self.model.generator,
                          update_rate=self.config['rollout_update_rate'], device=self.device)
        batch_size, seq_len = self.config['batch_size'], self.config['seq_len']
        for epoch in range(self.config['gan_epoch']):
            samples = self.model.generator.sample(batch_size, seq_len)
            rewards = rollout.get_reward(x=samples, num=self.config['rollout_num'],
                                         discriminator=self.model.discriminator)
            probs = self.model.generator(samples[:, :-1])
            # 原论文的 Mobility Regularity-Aware Loss 不带梯度, 这里仅使用 GAN Loss
            gan_loss = self.model.calc_gan_loss({
                'prob': probs.view(-1, probs.size(-1)),
                'action': samples[:, 1:].flatten(),
                'reward': rewards.flatten()
            })
            self.optimizers['gen_gan'].zero_grad()
            gan_loss.backward()
            self.optimizers['gen_gan'].step()

            rollout.update_params()
            dis_loss = self._train_dis_epoch(train_dataloader)

            self._logger.info(f'Adversarial train epoch {epoch} '
                              f'GAN loss: {gan_loss.item(): .4f}, Dis loss: {dis_loss: .4f}')

    def _gen_samples(self, batch_size, seq_len, gen_num):
        samples = []
        for _ in range(int(gen_num / batch_size)):
            sample = self.model.generator.sample(batch_size, seq_len).cpu().data.numpy().tolist()
            samples.extend(sample)
        return samples

    def _build_optimizer(self):
        return {
            'dis_pre': Adam(self.model.discriminator.parameters(), lr=self.config['dis_pre_lr']),
            'gen_pre': Adam(self.model.generator.parameters(), lr=self.config['gen_pre_lr']),
            # 'dis_gan': Adam(self.model.discriminator.parameters(), lr=self.config['dis_gan_lr']),
            'gen_gan': Adam(self.model.generator.parameters(), lr=self.config['gen_gan_lr'])
        }

    def evaluate(self, test_dataloader):
        """
        use model to test data

        Args:
            test_dataloader(torch.Dataloader): Dataloader
        """
        raise NotImplementedError("Executor evaluate not implemented")

    def load_model(self, cache_name):
        """
        加载对应模型的 cache

        Args:
            cache_name(str): 保存的文件名
        """
        raise NotImplementedError("Executor load cache not implemented")

    def save_model(self, cache_name):
        """
        将当前的模型保存到文件

        Args:
            cache_name(str): 保存的文件名
        """
        torch.save(self.model.state_dict(), cache_name)


class Rollout(object):
    """Roll-out policy"""
    def __init__(self, model, update_rate, device):
        self.optim_model = model
        self.rollout_model = copy.deepcopy(model)
        self.update_rate = update_rate
        self.device = device

    def get_reward(self, x, num, discriminator):
        """
        Args:
            x : (batch_size, seq_len) input data
            num : roll-out number
            discriminator : discrimanator model
        Returns:
            reward (batch_size, seq_len - 1)
        """
        batch_size, seq_len = x.size(0), x.size(1)
        rewards = torch.zeros_like(x).to(self.device)
        # 起点是直接采样得到的, 不需要 reward
        for step in range(2, seq_len + 1):
            obs_loc = x[:, 0: step]
            step_reward = torch.zeros(batch_size).to(self.device)
            for rol in range(num):
                samples = self.rollout_model.multi_step_pred(obs_loc, seq_len)
                pred = discriminator(samples)
                step_reward += pred
            rewards[:, step - 1] = step_reward / num
        return rewards[:, 1:]

    def update_params(self):
        dic = {}
        for name, param in self.optim_model.named_parameters():
            dic[name] = param.data
        for name, param in self.rollout_model.named_parameters():
            if name.startswith('emb') or name.startswith('Emb'):
                param.data = dic[name]
            else:
                param.data = self.update_rate * param.data + (1 - self.update_rate) * dic[name]