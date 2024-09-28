import ast
import math
import rtree
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from shapely.geometry import LineString, Point

from libcity.model.abstract_traj_gen_model import AbstractTrajectoryGenerationModel

# Reference: https://github.com/Yasoz/DiffTraj


class DiffTraj(AbstractTrajectoryGenerationModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.config = config
        self.mean = np.array([data_feature['mean_lon'], data_feature['mean_lat']])
        self.std = np.array([data_feature['std_lon'], data_feature['std_lat']])
        self.geo_rtree, self.geo_dict = self._build_rtree(data_feature['geo'])

        self.unet = UnetModel(config)
        self.device = config['device']
        self.n_steps = config['num_diffusion_timesteps']
        self.beta = torch.linspace(config['beta_start'], config['beta_end'], self.n_steps).to(self.device)
        self.traj_len = config['traj_len']
        self.alpha = 1. - self.beta
        self.eta = 0.0
        self.timesteps = 100
        self.skip = self.n_steps // self.timesteps
        self.seq = range(0, self.n_steps, self.skip)

    def generate(self, batch):
        """
        为了统一输出轨迹, 此处将生成的 GPS 轨迹转换为路段序列
        """
        batch_size = len(batch['gps'])
        x = torch.randn(batch_size, 2, self.traj_len).to(self.device)
        seq_next = [-1] + list(self.seq[:-1])
        for i, j in zip(reversed(self.seq), reversed(seq_next)):
            t = (torch.ones(batch_size) * i).to(self.device)
            next_t = (torch.ones(batch_size) * j).to(self.device)
            with torch.no_grad():
                pred_noise = self.unet(x, t)
                x = self._p_xt(x, pred_noise, t, next_t, self.beta, self.eta)
        norm_coords = x.transpose(1, 2).cpu().numpy()  # (batch_size, traj_len, 2)
        gps = norm_coords * self.std + self.mean

        gen_traj = []
        for i in range(len(gps)):
            loc_list = self._gps_to_loc(gps[i])
            gen_traj.append(loc_list)
        return gen_traj

    def _gps_to_loc(self, gps, d_lon=0.0005, d_lat=0.0005):
        """
        Args:
            gps: np.ndarray (traj_len, 2)
        Returns: List
        """
        loc_list = []
        for step in range(len(gps)):
            lon, lat = gps[step][0], gps[step][1]
            query_ids = list(self.geo_rtree.intersection((lon - d_lon, lat - d_lat,
                                                          lon + d_lon, lat + d_lat)))
            target_rid, min_dist = -1, math.inf
            for geo_id in query_ids:
                line = self.geo_dict[geo_id]
                distance = Point(lon, lat).distance(line)
                if distance < min_dist:
                    target_rid, min_dist = geo_id, distance
            if target_rid != -1:
                loc_list.append(target_rid)
        return self._merge_dup(loc_list)

    @staticmethod
    def _merge_dup(loc_list):
        if len(loc_list) == 0:
            return []
        result = [loc_list[0]]
        for num in loc_list[1:]:
            if num != result[-1]:
                result.append(num)
        return result

    @staticmethod
    def _build_rtree(geo_df):
        """
        构建 Rtree, 用于加速检索最近的路段
        """
        geom_dict = {
            row['geo_id']: LineString(ast.literal_eval(row['coordinates']))
            for _, row in geo_df.iterrows()
        }  # 安全解析坐标字符串
        geo_rtree = rtree.index.Index()
        for geo_id, line in geom_dict.items():
            geo_rtree.insert(id=geo_id, coordinates=line.bounds, obj=line)
        return geo_rtree, geom_dict

    def _p_xt(self, xt, noise, t, next_t, beta, eta):
        at = self._compute_alpha(beta, t.long())
        at_next = self._compute_alpha(beta, next_t.long())
        x0_t = (xt - noise * (1 - at).sqrt()) / at.sqrt()
        c1 = (eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt())
        c2 = ((1 - at_next) - c1 ** 2).sqrt()
        eps = torch.randn(xt.shape, device=xt.device)
        xt_next = at_next.sqrt() * x0_t + c1 * eps + c2 * noise
        return xt_next

    @staticmethod
    def _compute_alpha(beta, t):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1)
        return a

    def forward(self, batch, t):
        return self.unet(batch, t)


def get_timestep_embedding(timesteps, embedding_dim):
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class WideAndDeep(nn.Module):
    def __init__(self, config, embedding_dim=128, hidden_dim=256):
        super(WideAndDeep, self).__init__()

        # Wide part (linear model for continuous attributes)
        # self.wide_fc = nn.Linear(5, embedding_dim)
        self.wide_fc = nn.Linear(4, embedding_dim)

        # 离散特征: 星期几、出发路段 ID、达到路段 ID
        self.week_embedding = nn.Embedding(7, hidden_dim)
        self.sid_embedding = nn.Embedding(config.model.roads_num, hidden_dim)
        self.eid_embedding = nn.Embedding(config.model.roads_num, hidden_dim)
        self.deep_fc1 = nn.Linear(hidden_dim*3, embedding_dim)
        self.deep_fc2 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, attr):
        """
        attr: (batch_size, attr_num)
        """
        # Continuous attributes
        # continuous_attrs = attr[:, 1:6]
        continuous_attrs = attr[:, :4].float()

        # Categorical attributes
        # depature, sid, eid = attr[:, 0].long(
        # ), attr[:, 6].long(), attr[:, 7].long()
        sid, eid, week = attr[:, 4].long(), attr[:, 5].long(), attr[:, 6].long()

        # Wide part
        wide_out = self.wide_fc(continuous_attrs)

        # Deep part
        # depature_embed = self.depature_embedding(depature)
        week_embed = self.week_embedding(week)
        sid_embed = self.sid_embedding(sid)
        eid_embed = self.eid_embedding(eid)
        # categorical_embed = torch.cat(
        #     (depature_embed, sid_embed, eid_embed), dim=1)
        categorical_embed = torch.cat(
            (week_embed, sid_embed, eid_embed), dim=1)

        deep_out = F.relu(self.deep_fc1(categorical_embed))
        deep_out = self.deep_fc2(deep_out)
        # Combine wide and deep embeddings
        combined_embed = wide_out + deep_out

        return combined_embed


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32,
                              num_channels=in_channels,
                              eps=1e-6,
                              affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x,
                                            scale_factor=2.0,
                                            mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (1, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.1, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv1d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv1d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv1d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv1d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = h + self.temb_proj(nonlinearity(temb))[:, :, None]
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, w = q.shape
        q = q.permute(0, 2, 1)  # b,hw,c
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)
        # attend to values
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, w)

        h_ = self.proj_out(h_)

        return x + h_


class UnetModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        ch, out_ch, ch_mult = config['ch'], config['out_ch'], tuple(config['ch_mult'])
        self.attn_resolutions = config['attn_resolutions']
        self.dropout = config['dropout']
        resamp_with_conv = config['resamp_with_conv']
        num_timesteps = config['num_diffusion_timesteps']

        if config['type'] == 'bayesian':
            self.logvar = nn.Parameter(torch.zeros(num_timesteps))

        self.ch = config['ch']
        self.temb_ch = self.ch * 4
        self.num_resolutions = len(config['ch_mult'])
        self.num_res_blocks = config['num_res_blocks']
        self.resolution = config['traj_len']
        self.in_channels = config['in_channels']

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch, self.temb_ch),
            torch.nn.Linear(self.temb_ch, self.temb_ch),
        ])

        # downsampling
        self.conv_in = torch.nn.Conv1d(self.in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = self.resolution
        in_ch_mult = (1, ) + ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(in_channels=block_in,
                                out_channels=block_out,
                                temb_channels=self.temb_ch,
                                dropout=self.dropout))
                block_in = block_out
                if curr_res in self.attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                       temb_channels=self.temb_ch, dropout=self.dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                       temb_channels=self.temb_ch, dropout=self.dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(
                    ResnetBlock(in_channels=block_in + skip_in,
                                out_channels=block_out,
                                temb_channels=self.temb_ch,
                                dropout=self.dropout))
                block_in = block_out
                if curr_res in self.attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv1d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t, extra_embed=None):
        assert x.shape[2] == self.resolution

        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        if extra_embed is not None:
            temb = temb + extra_embed

        # downsampling
        hs = [self.conv_in(x)]
        # print(hs[-1].shape)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                # print(i_level, i_block, h.shape)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]  # [10, 256, 4, 4]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                ht = hs.pop()
                if ht.size(-1) != h.size(-1):
                    h = torch.nn.functional.pad(h, (0, ht.size(-1) - h.size(-1)))
                h = self.up[i_level].block[i_block](torch.cat([h, ht], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class GuideUNet(nn.Module):
    def __init__(self, config):
        super(GuideUNet, self).__init__()
        self.config = config
        self.ch = config.model.ch * 4
        self.attr_dim = config.model.attr_dim
        self.guidance_scale = config.model.guidance_scale
        self.unet = UnetModel(config)
        # self.guide_emb = Guide_Embedding(self.attr_dim, self.ch)
        # self.place_emb = Place_Embedding(self.attr_dim, self.ch)
        self.guide_emb = WideAndDeep(config, self.ch)
        self.place_emb = WideAndDeep(config, self.ch)

    def forward(self, x, t, attr):
        """
         x: (bat, 2, len)
         t: (bat)
         attr: (bat, 8)
        """
        guide_emb = self.guide_emb(attr)
        place_vector = torch.zeros(attr.shape, device=attr.device)
        place_emb = self.place_emb(place_vector)
        cond_noise = self.unet(x, t, guide_emb)
        uncond_noise = self.unet(x, t, place_emb)
        pred_noise = cond_noise + self.guidance_scale * (cond_noise -
                                                         uncond_noise)
        return pred_noise

