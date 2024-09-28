import os
import json
import time
import pyproj
import pandas as pd
import numpy as np
from scipy.stats import entropy

from libcity.evaluator.abstract_evaluator import AbstractEvaluator
from logging import getLogger


class TrajGenEvaluator(AbstractEvaluator):
    def __init__(self, config):
        self.config = config
        self.dataset = config.get('dataset', '')
        self.geo_file = config.get('geo_file', self.dataset)
        self.data_path = f'./raw_data/{self.dataset}/'

        # 位置 id 转坐标 (GPS 或其他)
        self.loc_to_lonlat = self._load_coords()
        self.latlon_to_xy = pyproj.Transformer.from_crs(4326, self.config['utm'])

        self.data_collector = {'real': [], 'gen': []}
        self.macro_metrics = ['Distance', 'Radius']
        self.result = dict()
        self._logger = getLogger()

    def _load_coords(self):
        geo_df = pd.read_csv(os.path.join(self.data_path, f'{self.geo_file}.geo'))
        geo_dict = {
            row['geo_id']: (row['lon'], row['lat'])
            for _, row in geo_df.iterrows()
        }
        return geo_dict

    def _calc_traj_attr(self, loc_list):
        loc_df = pd.DataFrame(data=loc_list, columns=['loc'])
        loc_df[['lon', 'lat']] = loc_df.apply(lambda row: self.loc_to_lonlat[row['loc']], axis=1, result_type='expand')
        loc_df['x'], loc_df['y'] = self.latlon_to_xy.transform(loc_df['lat'], loc_df['lon'])
        loc_df['Distance'] = np.linalg.norm([loc_df['x'].diff().fillna(0), loc_df['y'].diff().fillna(0)], axis=0)
        x_mean, y_mean = loc_df['x'].mean(), loc_df['y'].mean()
        loc_df['Radius'] = np.linalg.norm([loc_df['x'] - x_mean, loc_df['y'] - y_mean], axis=0)
        return {
            'Distance': loc_df['Distance'].sum() / 1000,
            'Radius': loc_df['Radius'].mean() / 1000
        }

    def _calc_macro_metrics(self, traj_df):
        traj_df['Distance'] = 0
        traj_df['Radius'] = 0
        for idx, row in traj_df.iterrows():
            attr = self._calc_traj_attr(row['loc_list'])
            traj_df.loc[idx, 'Distance'] = attr['Distance']
            traj_df.loc[idx, 'Radius'] = attr['Radius']
        return traj_df

    def _calc_loc_freq(self, traj_df):
        loc_freq = np.zeros(len(self.loc_to_lonlat))
        for _, row in traj_df.iterrows():
            for loc in row['loc_list']:
                loc_freq[loc] += 1
        loc_freq = loc_freq / loc_freq.sum()
        return loc_freq

    @staticmethod
    def _jsd(p, q):
        m = (p + q) / 2
        return 0.5 * entropy(p, m) + 0.5 * entropy(q, m)

    def collect(self, batch):
        """
        收集真实轨迹和生成轨迹
        Args:
        """
        if not isinstance(batch, dict):
            raise TypeError('evaluator.collect input is not a dict of user')
        self.data_collector['real'] += [(traj[0], traj) for idx, traj in enumerate(batch['real']) if len(traj) > 0]
        self.data_collector['gen'] += [(traj[0], traj) for idx, traj in enumerate(batch['gen']) if len(traj) > 0]

    def evaluate(self):
        real_df = pd.DataFrame(data=self.data_collector['real'], columns=['start_loc', 'loc_list'])
        gen_df = pd.DataFrame(data=self.data_collector['gen'], columns=['start_loc', 'loc_list'])
        real_df = self._calc_macro_metrics(real_df)
        gen_df = self._calc_macro_metrics(gen_df)
        # 轨迹的统计特征
        for metric in self.macro_metrics:
            min_val = min(real_df[metric].min(), gen_df[metric].min())
            max_val = max(real_df[metric].max(), gen_df[metric].max())
            real_hist, _ = np.histogram(real_df[metric], bins=50, range=(min_val, max_val), density=True)
            gen_hist, _ = np.histogram(gen_df[metric], bins=50, range=(min_val, max_val), density=True)
            self.result[metric] = self._jsd(real_hist, gen_hist)
        # 各位置的访问频率
        real_loc_freq = self._calc_loc_freq(real_df)
        gen_loc_freq = self._calc_loc_freq(gen_df)
        self.result['LocFreq'] = self._jsd(real_loc_freq, gen_loc_freq)
        return self.result

    def save_result(self, save_dir, filename=None):
        self.evaluate()
        if not os.path.exists(save_dir):
            # os.mkdir(save_path)
            os.makedirs(save_dir, exist_ok=True)
        if filename is None:
            # 使用时间戳
            filename = time.strftime(
                "%Y_%m_%d_%H_%M_%S", time.localtime(time.time()))
        self._logger.info('evaluate result is {}'.format(json.dumps(self.result, indent=1)))
        with open(os.path.join(save_dir, '{}.json'.format(filename)), 'w') as f:
            json.dump(self.result, f)

    def clear(self):
        self.result = {}
        self.data_collector = {'real': [], 'gen': []}
