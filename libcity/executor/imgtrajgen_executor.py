from ray import tune
import torch
import torch.optim as optim
import numpy as np
import os
from logging import getLogger
from torch.optim import Adam

from libcity.executor.abstract_executor import AbstractExecutor
from libcity.utils import get_evaluator


class ImgTrajGenExecutor(AbstractExecutor):
    def __init__(self, config, model, data_feature):
        # super().__init__()
        self.evaluator = get_evaluator(config)
        self.config = config
        self.data_feature = data_feature
        self.device = self.config['device']
        self.model = model.to(self.device)
        self.tmp_path = './libcity/tmp/checkpoint/'
        self.exp_id = self.config.get('exp_id', None)
        self.cache_dir = './libcity/cache/{}/model_cache'.format(self.exp_id)
        self.evaluate_res_dir = './libcity/cache/{}/evaluate_cache'.format(self.exp_id)
        self._logger = getLogger()
        self.optim_gen, self.optim_disc, self.optim_seq2seq = self._build_optimizer()

    def train(self, train_dataloader, eval_dataloader):
        """
        use data to train model with config

        Args:
            train_dataloader(torch.Dataloader): Dataloader
            eval_dataloader(torch.Dataloader): Dataloader
        """
        # Image Generation train
        for epoch in range(self.config['dcgan_epoch']):
            self._train_dcgan_epoch(train_dataloader)

        # Seq2Seq train
        for epoch in range(self.config['seq2seq_epoch']):
            self._train_seq2seq_epoch(train_dataloader)

    def _train_dcgan_epoch(self, train_dataloader):
        for i, batch in enumerate(train_dataloader):
            disc_loss = self.model.calculate_disc_loss(batch)
            self.optim_gen.zero_grad()
            disc_loss.backward()
            self.optim_gen.step()

            gen_loss = self.model.calculate_gen_loss(batch)
            self.optim_disc.zero_grad()
            gen_loss.backward()
            self.optim_disc.step()

            # if i % 100 == 0:
            #     print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
            #           % (i, len(train_dataloader), disc_loss.item(), gen_loss.item(), D_x, D_G_z1, D_G_z2))

    def _train_seq2seq_epoch(self, train_dataloader):
        avg_loss = 0
        for i, batch in enumerate(train_dataloader):
            seq2seq_loss = self.model.calculate_seq2seq_loss(batch)
            self.optim_seq2seq.zero_grad()
            seq2seq_loss.backward()
            self.optim_seq2seq.step()
            avg_loss += seq2seq_loss.item()
        return avg_loss / len(train_dataloader)

    def _build_optimizer(self):
        optim_gen = Adam(params=self.model.netG.parameters(), lr=self.config['lr'])
        optim_disc = Adam(params=self.model.netD.parameters(), lr=self.config['lr'])
        optim_seq2seq = Adam(params=self.model.seq2seq.parameters(), lr=self.config['lr'])
        return optim_gen, optim_disc, optim_seq2seq

    def evaluate(self, test_dataloader):
        """
        use model to test data

        Args:
            test_dataloader(torch.Dataloader): Dataloader
        """
        for batch in test_dataloader:
            real_batch = [
                [loc for loc in seq if loc != self.data_feature['loc_num']]
                for seq in batch['target_seq'].cpu().tolist()
            ]
            gen_batch = self.model.generate(batch)
            self.evaluator.collect({'real': real_batch, 'gen': gen_batch})
        self.evaluator.evaluate()
        self.evaluator.save_result(self.evaluate_res_dir)

    def load_model(self, cache_name):
        """
        加载对应模型的 cache

        Args:
            cache_name(str): 保存的文件名
        """
        model_state = torch.load(cache_name)
        self.model.load_state_dict(model_state)

    def save_model(self, cache_name):
        """
        将当前的模型保存到文件

        Args:
            cache_name(str): 保存的文件名
        """
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        torch.save(self.model.state_dict(), cache_name)