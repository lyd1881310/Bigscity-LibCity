import os
import math
import random
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from libcity.data.dataset import AbstractDataset
from libcity.data.list_dataset import ListDataset


class DiffTrajDataset(AbstractDataset):
    def __init__(self, config):
        self.config = config
        self.device = config['device']
        self.dataset = config.get('dataset', '')
        self.geo_file = config.get('geo_file', self.dataset)
        self.dyna_file = config.get('dyna_file', self.dataset)
        self.data_path = './raw_data/{}/'.format(self.dataset)
        self.geo_df = (pd.read_csv(os.path.join(self.data_path, f'{self.geo_file}.geo'))
                       .set_index('geo_id', drop=True))
        self.dyna_df = pd.read_csv(os.path.join(self.data_path, f'{self.dyna_file}.dyna'))
        self.traj_len = config['traj_len']
        self.data, self.mean_lon, self.mean_lat, self.std_lon, self.std_lat = self._preprocess()

    def get_data(self):
        train_num = math.ceil(self.data.shape[0] * self.config['train_rate'])
        eval_num = math.ceil(self.data.shape[0] * self.config['eval_rate'])
        train_data = self.data[:train_num, :, :]
        eval_data = self.data[train_num: train_num + eval_num, :, :]
        test_data = self.data[train_num + eval_num:, :, :]
        return self._get_dataloader(train_data), self._get_dataloader(eval_data), self._get_dataloader(test_data)

    def _get_dataloader(self, data):
        def collator(indices):
            return torch.from_numpy(np.stack(indices, axis=0)).to(self.device).swapdims(1, 2)
        dataset = ListDataset(data)
        return DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True, collate_fn=collator)
        # dataset = TensorDataset(torch.from_numpy(data).swapdims(1, 2))
        # return DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True,
        #                   collate_fn=lambda batch: batch.to(self.device))

    def _preprocess(self):
        gps_data = []
        for uid, traj in self.dyna_df.groupby('entity_id'):
            gps_list = []
            for loc_id in traj['location'].tolist():
                gps_list += eval(self.geo_df.loc[loc_id, "coordinates"])
            gps_arr = self._resample(gps_list)
            gps_data.append(gps_arr)
        random.shuffle(gps_data)
        gps_data = np.array(gps_data, dtype=np.float32)
        return self._normalize(gps_data)

    def _resample(self, gps_list):
        # 删除重复的 GPS 点
        x = [gps_list[0]]
        for item in gps_list[1:]:
            if item == x[-1]:
                continue
            x.append(item)
        x = np.array(x)
        # 插值
        len_x = len(x)
        time_steps = np.arange(self.traj_len) * (len_x - 1) / (self.traj_len - 1)
        x = x.T
        resampled = np.zeros((2, self.traj_len))
        for i in range(2):
            resampled[i] = np.interp(time_steps, np.arange(len_x), x[i])
        return resampled.T

    @staticmethod
    def _normalize(x: np.array):
        """
            x: (traj_num, length, 2)
        """
        x = x.copy()
        mean_lon, mean_lat = np.mean(x[:, :, 0]), np.mean(x[:, :, 1])
        std_lon, std_lat = np.std(x[:, :, 0]), np.std(x[:, :, 1])
        x[:, :, 0] = (x[:, :, 0] - mean_lon) / std_lon
        x[:, :, 1] = (x[:, :, 1] - mean_lat) / std_lat
        return x, mean_lon, mean_lat, std_lon, std_lat

    def get_data_feature(self):
        return {
            'mean_lon': self.mean_lon,
            'mean_lat': self.mean_lat,
            'std_lon': self.std_lon,
            'std_lat': self.std_lat
        }
