import os
import math
import random
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from libcity.data.dataset import AbstractDataset
from libcity.data.list_dataset import ListDataset


class MoveSimDataset(AbstractDataset):
    def __init__(self, config):
        self.config = config
        self.device = self.config['device']
        self.dataset = self.config.get('dataset', '')
        self.geo_file = self.config.get('geo_file', self.dataset)
        self.dyna_file = self.config.get('dyna_file', self.dataset)
        self.data_path = './raw_data/{}/'.format(self.dataset)
        self.geo_df = pd.read_csv(os.path.join(self.data_path, f'{self.geo_file}.geo'))
        self.dyna_df = pd.read_csv(os.path.join(self.data_path, f'{self.dyna_file}.dyna'))
        self.loc_num = len(self.geo_df)
        # 作者开源的代码将轨迹处理成定长序列
        self.seq_len = self.config['seq_len']

    def get_data(self):
        traj_list = []
        for uid, traj in self.dyna_df.groupby('entity_id'):
            loc_list = traj['location'].tolist()
            loc_list = loc_list * math.ceil(self.seq_len / len(loc_list))
            traj_list.append(loc_list[:self.seq_len])
        train_num = math.ceil(len(traj_list) * self.config['train_rate'])
        eval_num = math.ceil(len(traj_list) * self.config['eval_rate'])
        train_data = traj_list[:train_num]
        eval_data = traj_list[train_num: train_num + eval_num]
        test_data = traj_list[train_num + eval_num:]
        return self._get_dataloader(train_data), self._get_dataloader(eval_data), self._get_dataloader(test_data)

    def _get_dataloader(self, data_list):
        def collator(indices):
            return {'loc': torch.LongTensor(indices).to(self.device)}
        dataset = ListDataset(data_list)
        return DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True, collate_fn=collator)

    def _get_start_distri(self):
        start_distri = np.zeros(len(self.geo_df))
        for uid, traj in self.dyna_df.groupby('entity_id'):
            start_distri[traj.iloc[0]['location']] += 1
        return start_distri / start_distri.sum()

    def _get_trans_matrix(self):
        trans = np.zeros((self.loc_num, self.loc_num))
        for _, traj in tqdm(self.dyna_df.groupby('entity_id'), desc='get trans matrix'):
            for j in range(len(traj) - 1):
                trans[traj.iloc[j]['location'], traj.iloc[j + 1]['location']] += 1
        return trans

    def _get_dist_matrix(self):
        points = self.geo_df[['lon', 'lat']].to_numpy()
        diff_x = points[:, np.newaxis, 0] - points[np.newaxis, :, 0]
        diff_y = points[:, np.newaxis, 1] - points[np.newaxis, :, 1]
        squared_diff = diff_x ** 2 + diff_y ** 2
        return np.sqrt(squared_diff)

    @staticmethod
    def _distance(x1, y1, x2, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def get_data_feature(self):
        return {
            'loc_num': self.loc_num,
            'starting_dist': self._get_start_distri(),
            'M1': self._get_dist_matrix(),
            'M2': self._get_trans_matrix(),
            'M3': None
        }
    