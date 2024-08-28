import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from libcity.model.abstract_model import AbstractModel
from libcity.model.abstract_traj_gen_model import AbstractTrajectoryGenerationModel

# Reference: https://github.com/FIBLAB/MoveSim/tree/master


class MoveSim(AbstractTrajectoryGenerationModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.config = config
        self.tim_num = config['tim_num']
        self.device = config['device']
        self.data_feature = data_feature
        self.generator = ATGenerator(config, data_feature)
        self.discriminator = Discriminator(config, data_feature)

    def forward(self):
        pass

    def generate(self):
        pass

    def calc_dis_loss(self, batch):
        """
        Args:
            batch: Dict, {'real': torch.LongTensor, 'fake': torch.LongTensor}
        Returns:
        """
        real_label = torch.ones(batch['real'].shape[0]).to(self.device)
        fake_label = torch.zeros(batch['fake'].shape[0]).to(self.device)
        real_clf = self.discriminator(batch['real'])
        fake_clf = self.discriminator(batch['fake'])
        bce_loss = F.binary_cross_entropy(
            torch.cat([real_clf, fake_clf], dim=0),
            torch.cat([real_label, fake_label], dim=0)
        )
        return bce_loss

    def calc_gen_pre_loss(self, batch):
        intput = batch['loc'][:, :-1]
        target = batch['loc'][:, 1:]
        logit, label = [], []
        for i in range(intput.shape[1]):
            prob = self.generator(intput[:, :i+1])
            logit.append(prob)
            label.append(target[:, i])
        logit = torch.cat(logit, dim=0)
        label = torch.cat(label, dim=0)
        clf_loss = F.cross_entropy(logit, label)
        return clf_loss

    @staticmethod
    def calc_gan_loss(batch):
        """
        Args:
            batch: Dict {
                prob: (N, C), torch Variable
                action : (N, ), torch Variable
                reward : (N, ), torch Variable
            }
        """
        prob, action, reward = batch['prob'], batch['action'], batch['reward']
        act_prob = torch.gather(prob, dim=1, index=action.unsqueeze(-1)).squeeze(-1)
        loss = -torch.sum(act_prob * reward)
        return loss


class ATGenerator(nn.Module):
    """Attention Generator.
    """
    def __init__(self, config, data_feature):
        super(ATGenerator, self).__init__()

        self.loc_embedding_dim = config['loc_embedding_dim']
        self.tim_embedding_dim = config['tim_embedding_dim']
        self.embedding_dim = config['loc_embedding_dim'] + config['tim_embedding_dim']
        self.hidden_dim = config['hidden_dim']
        self.attn_layer_num = config['attn_layer_num']
        self.device = config['device']
        self.function = config['function']
        self.starting_sample = config['starting_sample']

        self.total_locations = data_feature['loc_num']
        self.M1 = torch.FloatTensor(data_feature['M1']).to(self.device)
        self.M2 = torch.FloatTensor(data_feature['M2']).to(self.device)
        self.M3 = torch.FloatTensor(data_feature['M3']).to(self.device) if self.function else None
        self.starting_dist = torch.FloatTensor(data_feature['starting_dist']).to(self.device)
        self.tim_num = config['tim_num']

        self.loc_embedding = nn.Embedding(
            num_embeddings=self.total_locations, embedding_dim=self.loc_embedding_dim)
        self.tim_embedding = nn.Embedding(
            num_embeddings=24, embedding_dim=self.tim_embedding_dim)

        self.attn_layers = nn.ModuleList([
            SelfAttention(input_dim=self.embedding_dim, hidden_dim=self.hidden_dim, num_heads=4),
            SelfAttention(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim, num_heads=1)
        ])

        self.ext_linear = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(self.total_locations, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Sigmoid(),
                nn.LayerNorm(self.hidden_dim)
            )
            for name in ['M1', 'M2', 'M3']
        })
        self.out_linear = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )
        self.pred_linear = nn.Sequential(
            nn.Linear(self.hidden_dim, self.total_locations),
            nn.Softmax(dim=-1)
        )
        self.init_params()

    def init_params(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)

    def forward(self, loc):
        """
        预测一步
        """
        # 时间片 Embedding 视为 Position Embedding
        batch_size, seq_len = loc.shape[0], loc.shape[1]
        tim = torch.LongTensor([i % self.tim_num for i in range(seq_len)]).repeat((batch_size, 1)).to(self.device)
        loc_emb = self.loc_embedding(loc)
        tim_emb = self.tim_embedding(tim)
        x = torch.cat([loc_emb, tim_emb], dim=-1)

        # Attention
        for attn_layer in self.attn_layers:
            x = attn_layer(x)
        x = self.out_linear(x)

        # Ext information
        mat1 = self.M1[loc]
        mat2 = self.M2[loc]
        mat1 = self.ext_linear['M1'](mat1)
        mat2 = self.ext_linear['M2'](mat2)
        # element-wise product
        if self.function:
            mat3 = self.M3[loc]
            mat3 = self.ext_linear['M3'](mat3)
            pred = self.pred_linear(x + x * (mat1 + mat2 + mat3))
        else:
            pred = self.pred_linear(x + x * (mat1 + mat2))
        # 取最后一步的预测结果
        return pred[:, -1, :]

    def multi_step_pred(self, obs_loc, seq_len):
        """
        Args:
            obs_loc:
            seq_len:
        Returns:
        """
        batch_size, obs_len = obs_loc.shape[0], obs_loc.shape[1]
        if obs_len >= seq_len:
            return obs_loc
        loc = obs_loc.to(self.device)
        for step in range(obs_len, seq_len):
            prob = self.forward(loc)
            nxt_loc = torch.multinomial(prob, 1)
            loc = torch.cat([loc, nxt_loc], dim=-1)
        return loc
        
    def sample(self, batch_size, seq_len):
        """
        从 0 开始生成一个 batch 的定长轨迹
        Returns:
        """
        # sample start location
        loc = torch.cat(
            [torch.multinomial(self.starting_dist, 1) for _ in range(batch_size)],
            dim=-1
        ).unsqueeze(-1).to(self.device)

        # auto regression
        for step in range(1, seq_len):
            prob = self.forward(loc)
            nxt_loc = torch.multinomial(prob, 1)
            loc = torch.cat([loc, nxt_loc], dim=-1)
        return loc


class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.Q = nn.Linear(input_dim, hidden_dim)
        self.V = nn.Linear(input_dim, hidden_dim)
        self.K = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        q = torch.relu(self.Q(x))
        k = torch.relu(self.K(x))
        v = torch.relu(self.V(x))
        x, _ = self.attn(q, k, v)
        return x


class Discriminator(nn.Module):
    """
    Basic discriminator.
    """
    def __init__(self, config, data_feature):
        super(Discriminator, self).__init__()
        num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
        filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
        self.embedding = nn.Embedding(num_embeddings=data_feature['loc_num'],
                                      embedding_dim=config['disc_embedding_dim'])
        self.convs = nn.ModuleList([nn.Conv2d(1, n, (f, config['disc_embedding_dim']))
                                    for (n, f) in zip(num_filters, filter_sizes)])
        self.highway = nn.Linear(sum(num_filters), sum(num_filters))
        self.dropout = nn.Dropout(p=config['dropout'])
        #
        self.pred_mlp = nn.Sequential(
            nn.Linear(sum(num_filters), 1),
            nn.Sigmoid()
        )
        self.init_parameters()

    def forward(self, x):
        """
        Args:
            x: (batch_size * seq_len)
        """
        emb = self.embedding(x).unsqueeze(1)  # batch_size * 1 * seq_len * emb_dim
        # [batch_size * num_filter * length]
        convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2)
                 for conv in convs]  # [batch_size * num_filter]
        pred = torch.cat(pools, 1)  # batch_size * num_filters_sum
        highway = self.highway(pred)
        pred = torch.sigmoid(highway) * F.relu(highway) + \
            (1. - torch.sigmoid(highway)) * pred
        # pred = F.log_softmax(self.pred_mlp(self.dropout(pred)), dim=-1)
        pred = self.pred_mlp(self.dropout(pred)).squeeze()
        return pred

    def init_parameters(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)