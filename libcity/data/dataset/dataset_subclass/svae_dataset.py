import os
import math
import json
import random
import pandas as pd
from libcity.data.dataset import AbstractDataset
from libcity.data.utils import ListDataset

from libcity.data.dataset import AbstractDataset
from libcity.utils import parse_time, cal_timeoff
from libcity.data.utils import generate_dataloader_pad


class SVAEDataset(AbstractDataset):
    def __init__(self, config):
        # super().__init__(config)
        self.config = config
        self.cache_file_folder = './libcity/cache/dataset_cache/'
        self.dataset = self.config.get('dataset', '')
        self.geo_file = self.config.get('geo_file', self.dataset)
        self.dyna_file = self.config.get('dyna_file', self.dataset)
        self.data_path = './raw_data/{}/'.format(self.dataset)
        self.data = None
        # 加载 encoder
        # self.encoder = self.get_encoder()
        # self.pad_item = None  # 因为若是使用缓存, pad_item 是记录在缓存文件中的而不是 encoder
        # self.logger = getLogger()

    def get_data(self):
        if self.data is not None:
            return
        geo_df = pd.read_csv(os.path.join(self.data_path, '{}.geo'.format(self.geo_file)))
        self.pad_item = geo_df['geo_id'].max() + 1
        self.data = []
        traj_df = pd.read_csv(os.path.join(self.data_path, '{}.dyna'.format(self.dyna_file)))
        for uid, traj in traj_df.groupby('entity_id'):
            self.data.append([traj['location'].tolist(), len(traj['location'])])
        random.shuffle(self.data)
        train_num = math.ceil(len(self.data) * self.config['train_rate'])
        eval_num = math.ceil(len(self.data) * self.config['eval_rate'])
        train_data = self.data[:train_num]
        eval_data = self.data[train_num: train_num + eval_num]
        test_data = self.data[train_num + eval_num:]

        return generate_dataloader_pad(
            train_data=train_data, eval_data=eval_data, test_data=test_data,
            feature_name={'seq': 'int', 'length': 'no_tensor'}, batch_size=self.config['batch_size'], num_workers=self.config['num_workers'],
            pad_item={'seq': self.pad_item}
        )

    def get_data_feature(self):
        return {
            'pad_item': self.pad_item
        }
