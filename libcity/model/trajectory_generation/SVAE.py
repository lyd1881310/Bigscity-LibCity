import torch
import torch.nn as nn
import random
from libcity.model.abstract_model import AbstractModel


class SVAE(AbstractModel):
    """
    Variational Auto Encoder
    """
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.config = config
        self.data_feature = data_feature
        self.input_size = data_feature['pad_item'] + 1
        self.hidden_size = config['hidden_size']
        self.z_size = config['z_size']
        self.n_layers = config['n_layers']
        self.dropout = config['dropout']
        self.teacher_forcing_ratio = config['teacher_forcing_ratio']
        self.kl_weight = config['kl_weight']
        self.device = config['device']

        self.pad_token = data_feature['pad_item']

        self.encoder = Encoder(input_size=self.input_size, hidden_size=self.hidden_size, device=self.device,
                               n_layers=self.n_layers, z_size=self.z_size, road_pad=self.pad_token)
        self.decoder = Decoder(input_size=self.input_size, hidden_size=self.hidden_size, device=self.device,
                               n_layers=self.n_layers, z_size=self.z_size, road_pad=self.pad_token,
                               teacher_forcing_ratio=self.teacher_forcing_ratio)

    def calculate_loss(self, batch):
        input_seqs = batch['seq']
        input_lengths = batch['length']
        batch_size = input_seqs.shape[0]
        z_mean, z_log_var = self.encoder(input_seqs, input_lengths)
        std = z_log_var.mul(0.5).exp_()
        # sampling epsilon from normal distribution
        epsilon = torch.randn(batch_size, self.z_size).to(self.device)
        z = z_mean + std * epsilon
        # encoder-decoder loss
        decoder_loss = self.decoder.calculate_loss(z, input_lengths=input_lengths, target_seq=input_seqs)
        # KL loss N(mean, var) with N(0, 1)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + z_log_var - z_mean ** 2 - z_log_var.exp(), dim=1), dim=0)
        return decoder_loss + self.kl_weight * kld_loss

    def generate(self, batch):
        traj_num = batch['seq'].shape[0]
        z = torch.randn(traj_num, self.z_size).to(self.device)
        sample_seq = self.decoder.decode(z, batch['length'])
        return sample_seq

    # def sample(self, sample_nums, input_lengths):
    #     z = torch.randn(sample_nums, self.z_size).to(self.device)
    #     sample_seq = self.decoder.decode(z, input_lengths)
    #     return sample_seq
    #
    def reconstruct(self, batch):
        input_seqs = batch['seq']
        input_lengths = batch['length']
        batch_size = input_seqs.shape[0]
        z_mean, z_log_var = self.encoder(input_seqs, input_lengths)
        std = z_log_var.mul(0.5).exp_()
        epsilon = torch.randn(batch_size, self.z_size).to(self.device)
        z = z_mean + std * epsilon
        reconstruct_seq = self.decoder.decode(z, input_lengths)
        return reconstruct_seq


class Encoder(nn.Module):
    """
    the encoder of seq2seq model
    add mean mapping function and log variance mapping function
    """

    def __init__(self, input_size, hidden_size, device, z_size, n_layers=1, road_pad=0):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.n_layers = n_layers
        self.z_size = z_size
        self.road_pad = road_pad

        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=self.road_pad)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers)
        self.fc_mean = nn.Linear(self.hidden_size, self.z_size)
        self.fc_log_var = nn.Linear(self.hidden_size, self.z_size)

    def forward(self, input_seqs, input_lengths):
        embedded = self.embedding(input_seqs)
        if input_seqs.shape[0] != 1:
            packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths=input_lengths, batch_first=True,
                                                             enforce_sorted=False)
            outputs, (h_n, c_n) = self.lstm(packed)
            outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
            outputs = outputs.transpose(0, 1)
            # get final output
            final_out_index = torch.tensor(input_lengths) - 1
            final_out_index = final_out_index.reshape(final_out_index.shape[0], 1, -1)
            final_out_index = final_out_index.repeat(1, 1, self.hidden_size).to(self.device)
            outputs = torch.gather(outputs, 1, final_out_index).squeeze(1)  # batch_size * hidden_size
        else:
            outputs, (h_n, c_n) = self.lstm(embedded)
            outputs = outputs[:, -1].squeeze(1)
        # map outputs to u and sigma
        mean = self.fc_mean(outputs)  # batch_size * z_size
        log_var = self.fc_log_var(outputs)  # batch_size * z_size
        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, device, z_size, n_layers=1, road_pad=0, teacher_forcing_ratio=0.8):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.n_layers = n_layers
        self.z_size = z_size
        self.road_pad = road_pad
        self.teacher_forcing_rate = teacher_forcing_ratio

        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=self.road_pad)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers)
        self.out = nn.Linear(self.hidden_size, self.input_size)
        self.decode_h0 = nn.Linear(self.z_size, self.hidden_size * self.n_layers)
        self.decode_c0 = nn.Linear(self.z_size, self.hidden_size * self.n_layers)
        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=road_pad).to(device)

    def forward(self, input_seq, h_0, c_0):
        embedded = self.embedding(input_seq)
        # 因为默认 GRU 是 batch_first False 的
        embedded = embedded.unsqueeze(0)

        rnn_output, (h_n, c_n) = self.lstm(embedded, (h_0, c_0))

        rnn_output = rnn_output.squeeze(0)

        output = self.out(rnn_output)

        return output, h_n, c_n

    def calculate_loss(self, z, input_lengths, target_seq):
        """
        reconstruct input sequence
        Args:
            z: the latent vector, encode from encoder
            input_lengths: 输入序列的长度
            target_seq: 目标序列

        Returns:
            decoder_loss
        """
        batch_size = z.shape[0]
        decoder_input = torch.LongTensor([self.road_pad] * batch_size).to(self.device)
        h_0 = self.decode_h0(z).unsqueeze(0)
        c_0 = self.decode_c0(z).unsqueeze(0)

        max_length = max(input_lengths)
        all_decoder_output = torch.zeros((max_length, batch_size, self.input_size)).to(self.device)
        target_seq = target_seq.transpose(0, 1)  # 使得 seq_len 在第一维度
        for t in range(max_length):
            decoder_output, h_n, c_n = self.forward(decoder_input, h_0, c_0)
            all_decoder_output[t] = decoder_output
            use_teacher_forcing = True if random.random() < self.teacher_forcing_rate else False
            if use_teacher_forcing:
                decoder_input = target_seq[t]  # (batch_size)
            else:
                # 选取 decoder_output 中概率最大的点作为 decode 的值
                val, index = torch.topk(decoder_output.detach(), 1, dim=1)
                decoder_input = index.squeeze(1)  # (batch_size)
            h_0 = h_n
            c_0 = c_n
        loss = self.loss_func(all_decoder_output.transpose(0, 1).contiguous().view(batch_size * max_length, -1),
                              target_seq.transpose(0, 1).contiguous().view(batch_size * max_length))
        return loss

    def decode(self, z, input_lengths):
        batch_size = z.shape[0]
        decoder_input = torch.LongTensor([self.road_pad] * batch_size).to(self.device)
        h_0 = self.decode_h0(z).unsqueeze(0)
        c_0 = self.decode_c0(z).unsqueeze(0)

        sample_seq = []
        max_length = max(input_lengths)
        for t in range(max_length):
            decoder_output, h_n, c_n = self.forward(decoder_input, h_0, c_0)
            # 选取 decoder_output 中概率最大的点作为 decode 的值
            val, index = torch.topk(decoder_output.detach(), 1, dim=1)
            decoder_input = index.squeeze(1)  # (batch_size)
            h_0 = h_n
            c_0 = c_n
            sample_seq.append(decoder_input.unsqueeze(1))
        sample_seq = torch.cat(sample_seq, dim=1)
        return sample_seq

