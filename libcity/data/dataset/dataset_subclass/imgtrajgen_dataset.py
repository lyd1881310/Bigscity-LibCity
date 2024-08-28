import os
import math
import random
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from libcity.data.dataset import AbstractDataset
from libcity.data.list_dataset import ListDataset
from libcity.data.batch import Batch, BatchPAD


class ImgTrajGenDataset(AbstractDataset):
    def __init__(self, config):
        self.config = config
        self.device = config['device']
        self.dataset = self.config.get('dataset', '')
        self.geo_file = self.config.get('geo_file', self.dataset)
        self.dyna_file = self.config.get('dyna_file', self.dataset)
        self.data_path = './raw_data/{}/'.format(self.dataset)
        self.geo_df = pd.read_csv(os.path.join(self.data_path, f'{self.geo_file}.geo'))
        self.pad_token = len(self.geo_df)
        self.dyna_df = pd.read_csv(os.path.join(self.data_path, f'{self.dyna_file}.dyna'))
        self.data_trans = self._build_data_trans()
        self.data = []

    def _build_data_trans(self):
        loc_to_gps = {
            row['geo_id']: (row['lon'], row['lat'])
            for _, row in self.geo_df.iterrows()
        }
        return DataTrans(config=self.config, rid_gps=loc_to_gps)

    def _collator(self, indices):
        sorted_list = [torch.LongTensor(seq) for seq, _, _ in indices]
        true_list = [torch.LongTensor(trg) for _, _, trg in indices]
        seq_length = [len(trg) for _, _, trg in indices]
        input_seq = pad_sequence(sorted_list, batch_first=True, padding_value=self.pad_token)
        target_seq = pad_sequence(true_list, batch_first=True, padding_value=self.pad_token)
        img_batch = torch.cat([img.unsqueeze(0) for _, img, _ in indices], dim=0)
        return {
            'img': img_batch.to(self.device),
            'input_seq': input_seq.to(self.device),
            'seq_length': seq_length,
            'target_seq': target_seq.to(self.device)
        }

    def _get_dataloader(self, data_list):
        dataset = ListDataset(data_list)
        return DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True, collate_fn=self._collator)

    def get_data(self):
        """
        返回数据的DataLoader，包括训练数据、测试数据、验证数据

        Returns:
            tuple: tuple contains:
                train_dataloader: Dataloader composed of Batch (class) \n
                eval_dataloader: Dataloader composed of Batch (class) \n
                test_dataloader: Dataloader composed of Batch (class)
        """
        for uid, traj in self.dyna_df.groupby('entity_id'):
            loc_list = traj['location'].tolist()
            sort_loc_list = sorted(loc_list)
            img = self.data_trans.traj_to_img(loc_list)
            self.data.append((sort_loc_list, img, loc_list))
        random.shuffle(self.data)
        train_num = math.ceil(len(self.data) * self.config['train_rate'])
        eval_num = math.ceil(len(self.data) * self.config['eval_rate'])
        train_data = self.data[:train_num]
        eval_data = self.data[train_num: train_num + eval_num]
        test_data = self.data[train_num + eval_num:]

        return self._get_dataloader(train_data), self._get_dataloader(eval_data), self._get_dataloader(test_data)

    def get_data_feature(self):
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {
            'loc_num': len(self.geo_df),
            'img_width': self.data_trans.img_width,
            'img_height': self.data_trans.img_height
        }


class DataTrans:
    def __init__(self, config, rid_gps):
        self.config = config
        self.rid_gps = rid_gps
        self.img_unit = config['img_unit']
        lon_coords = [lon for _, (lon, lat) in rid_gps.items()]
        lat_coords = [lat for _, (lon, lat) in rid_gps.items()]

        img_eps = config['img_eps']
        self.lon_0 = min(lon_coords) - img_eps
        self.lon_1 = max(lon_coords) + img_eps
        self.lat_0 = min(lat_coords) - img_eps
        self.lat_1 = max(lat_coords) + img_eps
        self.img_width = math.ceil((self.lon_1 - self.lon_0) / self.img_unit) + 1  # 图像的宽度
        self.img_height = math.ceil((self.lat_1 - self.lat_0) / self.img_unit) + 1  # 映射出的图像的高度

        # 网格 -> 路段 ID 的映射
        self.grid_road_dict = {x: {y: [] for y in range(self.img_height)}
                               for x in range(self.img_width)}
        for road, (lon, lat) in self.rid_gps.items():
            x_, y_ = self.gps_to_grid(lon, lat)
            self.grid_road_dict[x_][y_].append(int(road))

        self.transform = transforms.Compose([
            transforms.ToPILImage(mode='L'),
            transforms.Resize(self.config['imsize']),
            transforms.CenterCrop(self.config['imsize']),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))]
        )

    def gps_to_grid(self, lon, lat):
        """
        GPS 经纬度点映射为图像网格
        Args:
            lon: 经度
            lat: 纬度
        Returns:
            x, y: 映射的网格的 x 与 y 坐标
        """
        assert self.lon_0 <= lon <= self.lon_1 and self.lat_0 <= lat <= self.lat_1, 'GPS Coords Out of Range'
        x = math.floor((lon - self.lon_0) / self.img_unit)
        y = math.floor((lat - self.lat_0) / self.img_unit)
        return x, y

    def grid_to_gps(self, x, y):
        """
        图像网格还原为 GPS 经纬度点
        Args:
            x: 网格横坐标
            y: 网格纵坐标
        Returns:
            lon, lat
        """
        assert 0 <= x <= self.img_width and 0 <= y <= self.img_height, 'Grid Coords Out of Range'
        # 取这个网格区域的中心点
        lon = self.lon_0 + x * self.img_unit + self.img_unit / 2
        lat = self.lat_0 + y * self.img_unit + self.img_unit / 2
        return lon, lat

    def grid_to_loc(self, x, y):
        """
        直接把网格随机映射到这个网格里的路段上
        Args:
            x: 网格横坐标
            y: 网格纵坐标
        Returns:
            road id
        """
        road_set = self.grid_road_dict[x][y]
        if len(road_set) == 0:
            return None
        return np.random.choice(road_set)

    def gps_to_img(self, trace):
        """
        GPS 轨迹转成黑白图像
        Args:
            trace: GPS 数组，即一条轨迹的 GPS
        Returns:
            img (numpy.array)： H * W * 1 的二值图像
        """
        # H * W * C C 表示信道数，因为是二值图像，所以 C = 1
        img = np.ones((self.img_height, self.img_width, 1), dtype=np.uint8)  # 255 为白色，0 为黑色
        img = img * 255  # 初始画布为全白
        dxy = [(-1, -1), (-1, 0), (-1, 1),
               (0, -1), (0, 0), (0, 1),
               (1, -1), (1, 0), (1, 1)]

        for point in trace:
            x, y = self.gps_to_grid(point[0], point[1])
            # 把周边九宫格标记成黑色
            for dx, dy in dxy:
                nx, ny = x + dx, y + dy
                if nx < 0 or nx >= self.img_width or ny < 0 or ny >= self.img_height:
                    continue
                # 注意: y 代表行数, x 代表列数
                img[ny][nx] = 0
        return img

    def img_to_traj(self, img):
        """
        图像转成 路段 序列
        Args:
            img (tensor)： H * W 的图像，取值范围 [-1, 1] => [0, 255]
        Returns:
            trace (list): road 数组，即一条轨迹的 GPS
        """
        # 视小于 0 的点为黑色的轨迹点
        trace = []
        # img_np = img[0].numpy()
        img_np = img.numpy()
        x_list, y_list = np.where(img_np < 0)
        for i in range(x_list.shape[0]):
            # HACK: 行列和横纵坐标是反过来对应的
            x_i, y_i = y_list[i], x_list[i]
            road = self.grid_to_loc(x_i, y_i)
            if road is not None:
                trace.append(road)
        return trace

    def traj_to_img(self, loc_list):
        gps_list = [self.rid_gps[loc] for loc in loc_list]
        bi_img = self.gps_to_img(gps_list)
        return self.transform(bi_img)

