from ray import tune
import torch
import torch.optim as optim
import numpy as np
import os
from logging import getLogger

from libcity.executor.abstract_executor import AbstractExecutor
from libcity.utils import get_evaluator


class SVAEExecutor(AbstractExecutor):

    def __init__(self, config, model, data_feature):
        self.evaluator = get_evaluator(config)
        self.config = config
        self.model = model.to(self.config['device'])
        self.tmp_path = './libcity/tmp/checkpoint/'
        self.exp_id = self.config.get('exp_id', None)
        self.cache_dir = './libcity/cache/{}/model_cache'.format(self.exp_id)
        self.evaluate_res_dir = './libcity/cache/{}/evaluate_cache'.format(self.exp_id)
        self._logger = getLogger()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

    def train(self, train_dataloader, eval_dataloader):
        if not os.path.exists(self.tmp_path):
            os.makedirs(self.tmp_path)
        metrics = {}
        metrics['accuracy'] = []
        metrics['loss'] = []
        lr = self.config['learning_rate']
        from tqdm import tqdm
        for epoch in tqdm(range(self.config['max_epoch'])):
            self._logger.info('start train')
            self.model, avg_loss = self._train_epoch(train_dataloader, self.model, self.config['clip'])
            self._logger.info('==>Train Epoch:{:4d} Loss:{:.5f} learning_rate:{}'.format(
                epoch, avg_loss, lr))
            # eval stage
            self._logger.info('start evaluate')
            avg_eval_acc = self._valid_epoch(eval_dataloader)
            self._logger.info('==>Eval Acc:{:.5f}'.format(avg_eval_acc))

            lr = self.optimizer.param_groups[0]['lr']
            if lr < self.config['early_stop_lr']:
                break
        self.save_model(os.path.join(self.cache_dir, 'svae.pth'))
        # if not self.config['hyper_tune'] and self.config['load_best_epoch']:
        #     best = np.argmax(metrics['accuracy'])  # 这个不是最好的一次吗？
        #     load_name_tmp = 'ep_' + str(best) + '.m'
        #     self.model.load_state_dict(
        #         torch.load(self.tmp_path + load_name_tmp))
        # # 删除之前创建的临时文件夹
        # for rt, dirs, files in os.walk(self.tmp_path):
        #     for name in files:
        #         remove_path = os.path.join(rt, name)
        #         os.remove(remove_path)
        # os.rmdir(self.tmp_path)

    def load_model(self, cache_name):
        model_state, optimizer_state = torch.load(cache_name)
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)

    def save_model(self, cache_name):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        # save optimizer when load epoch to train
        torch.save((self.model.state_dict(), self.optimizer.state_dict()), cache_name)

    def evaluate(self, test_dataloader):
        self.model.train(False)
        self.evaluator.clear()
        for batch in test_dataloader:
            batch.to_tensor(device=self.config['device'])
            gen_batch = self.model.generate(batch)
            real_list = self._to_list_traj(batch['seq'])
            gen_list = self._to_list_traj(gen_batch)
            eval_input = {
                'real': real_list,
                'gen': gen_list
            }
            self.evaluator.collect(eval_input)
        self.evaluator.evaluate()
        self.evaluator.save_result(self.evaluate_res_dir)

    def _to_list_traj(self, traj_batch):
        """
        Args:
            traj: (batch_size, max_length)
        Returns:
        """
        list_batch = []
        pad_token = self.model.pad_token
        for traj in traj_batch.detach().cpu().tolist():
            traj_list = []
            for loc in traj:
                if loc == pad_token:
                    break
                traj_list.append(loc)
            list_batch.append(traj_list)
        return list_batch

    def _train_epoch(self, data_loader, model, clip):
        model.train(True)
        if self.config['debug']:
            torch.autograd.set_detect_anomaly(True)
        total_loss = []
        for batch in data_loader:
            # one batch, one step
            batch.to_tensor(device=self.config['device'])
            loss = model.calculate_loss(batch)
            self.optimizer.zero_grad()
            loss.backward()
            total_loss.append(loss.data.cpu().numpy().tolist())
            try:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            except:
                pass
            self.optimizer.step()
        avg_loss = np.mean(total_loss, dtype=np.float64)
        return model, avg_loss

    def _valid_epoch(self, data_loader):
        total_hit, total_cnt = 0, 0
        for batch in data_loader:
            batch.to_tensor(self.config['device'])
            input_seq, seq_length = batch['seq'], batch['length']
            recons_seq = self.model.reconstruct(batch)
            for i in range(recons_seq.shape[0]):
                for j in range(seq_length[i]):
                    if input_seq[i][j].item() == self.model.pad_token:
                        continue
                    elif recons_seq[i][j].item() == input_seq[i][j].item():
                        total_hit += 1
                    total_cnt += 1
        return total_hit / (total_cnt + 1e-3)

    def _build_optimizer(self):
        """
        根据全局参数`learner`选择optimizer
        """
        if self.config['optimizer'] == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'],
                                   weight_decay=self.config['L2'])
        elif self.config['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config['learning_rate'],
                                        weight_decay=self.config['L2'])
        elif self.config['optimizer'] == 'adagrad':
            optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.config['learning_rate'],
                                            weight_decay=self.config['L2'])
        elif self.config['optimizer'] == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.config['learning_rate'],
                                            weight_decay=self.config['L2'])
        elif self.config['optimizer'] == 'sparse_adam':
            optimizer = torch.optim.SparseAdam(self.model.parameters(), lr=self.config['learning_rate'])
        else:
            self._logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'],
                                   weight_decay=self.config['L2'])
        return optimizer

    def _build_scheduler(self):
        """
        目前就固定的 scheduler 吧
        """
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max',
                                                         patience=self.config['lr_step'],
                                                         factor=self.config['lr_decay'],
                                                         threshold=self.config['schedule_threshold'])
        return scheduler
