import torch
import torch.nn as nn
import torch.nn.functional as F

from libcity.model.abstract_model import AbstractModel

# Reference: https://github.com/Natsu6767/DCGAN-PyTorch/blob/d3165b562cca2c03f9b0c512ddedd81d23833363/dcgan.py


class ImgTrajGen(AbstractModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.config = config
        self.netG = Generator(config)
        self.netD = Discriminator(config)
        self.seq2seq = Seq2Seq(config, data_feature)
        self.real_label = 1.0
        self.fake_label = 0.0
        self.device = config['device']
        self.disc_loss = nn.BCELoss()
        self.seq2seq_loss = nn.CrossEntropyLoss(ignore_index=data_feature['loc_num'])

    def generate_img(self, batch):
        b_size = batch['img'].size(0)
        noise = torch.randn(b_size, self.config['nz'], 1, 1, device=self.device)
        fake_data = self.netG(noise)
        return fake_data

    def calculate_disc_loss(self, batch):
        real_data = batch['img']
        b_size = real_data.size(0)
        real_label = torch.full((b_size,), self.real_label, device=self.device)
        output = self.netD(real_data).view(-1)
        disc_loss_real = self.disc_loss(output, real_label)

        fake_data = self.generate_img(batch)
        fake_label = torch.full((b_size,), self.fake_label, device=self.device)
        output = self.netD(fake_data.detach()).view(-1)
        disc_loss_fake = self.disc_loss(output, fake_label)
        disc_loss = disc_loss_real + disc_loss_fake
        return disc_loss

    def calculate_gen_loss(self, batch):
        b_size = batch['img'].size(0)
        label = torch.full((b_size,), self.real_label, device=self.device)
        fake_data = self.generate_img(batch)
        output = self.netD(fake_data).view(-1)
        gen_loss = self.disc_loss(output, label)
        return gen_loss

    def calculate_seq2seq_loss(self, batch):
        decode_output = self.seq2seq(batch)
        return self.seq2seq_loss(decode_output, batch['target_seq'])

    def generate(self, batch, data_trans):
        gen_img = self.generate_img(batch)
        traj_list = []
        for i in range(gen_img.size(0)):
            loc_list = data_trans.img_to_traj(gen_img[i])
            traj_list.append(loc_list)
        return traj_list


def weights_init(w):
    """
    Initializes the weights of the layer, w.
    """
    classname = w.__class__.__name__
    if classname.find('conv') != -1:
        nn.init.normal_(w.weight.data, 0.0, 0.02)
    elif classname.find('bn') != -1:
        nn.init.normal_(w.weight.data, 1.0, 0.02)
        nn.init.constant_(w.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Input is the latent vector Z.
        self.tconv1 = nn.ConvTranspose2d(config['nz'], config['ngf'] * 8,
                                         kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(config['ngf'] * 8)

        # Input Dimension: (ngf*8) x 4 x 4
        self.tconv2 = nn.ConvTranspose2d(config['ngf'] * 8, config['ngf'] * 4,
                                         4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(config['ngf'] * 4)

        # Input Dimension: (ngf*4) x 8 x 8
        self.tconv3 = nn.ConvTranspose2d(config['ngf'] * 4, config['ngf'] * 2,
                                         4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(config['ngf'] * 2)

        # Input Dimension: (ngf*2) x 16 x 16
        self.tconv4 = nn.ConvTranspose2d(config['ngf'] * 2, config['ngf'],
                                         4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(config['ngf'])

        # Input Dimension: (ngf) * 32 * 32
        self.tconv5 = nn.ConvTranspose2d(config['ngf'], config['nc'],
                                         4, 2, 1, bias=False)
        #Output Dimension: (nc) x 64 x 64

    def forward(self, x):
        x = F.relu(self.bn1(self.tconv1(x)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.bn3(self.tconv3(x)))
        x = F.relu(self.bn4(self.tconv4(x)))

        x = torch.tanh(self.tconv5(x))

        return x


# Define the Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Input Dimension: (nc) x 64 x 64
        self.conv1 = nn.Conv2d(config['nc'], config['ndf'],
                               4, 2, 1, bias=False)

        # Input Dimension: (ndf) x 32 x 32
        self.conv2 = nn.Conv2d(config['ndf'], config['ndf'] * 2,
                               4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(config['ndf'] * 2)

        # Input Dimension: (ndf*2) x 16 x 16
        self.conv3 = nn.Conv2d(config['ndf'] * 2, config['ndf'] * 4,
                               4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(config['ndf'] * 4)

        # Input Dimension: (ndf*4) x 8 x 8
        self.conv4 = nn.Conv2d(config['ndf'] * 4, config['ndf'] * 8,
                               4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(config['ndf'] * 8)

        # Input Dimension: (ndf*8) x 4 x 4
        self.conv5 = nn.Conv2d(config['ndf'] * 8, 1, 4, 1, 0, bias=False)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2, True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2, True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2, True)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2, True)

        x = torch.sigmoid(self.conv5(x))

        return x


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        embedded = self.embedding(input_seqs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths=input_lengths, batch_first=True, enforce_sorted=False)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs) # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        return outputs, hidden


class Attn(nn.Module):
    def __init__(self, method, hidden_size, device):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size
        self.device = device

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)

        attn_energies = torch.zeros(this_batch_size, max_len).to(self.device)  # B x S

        for b in range(this_batch_size):
            for i in range(max_len):
                attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0))

        return torch.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_output):
        if self.method == 'dot':
            energy = torch.dot(hidden.view(-1), encoder_output.view(-1))
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = torch.dot(hidden.view(-1), energy.view(-1))
        else:
            assert self.method == 'concat'
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = torch.dot(self.v.view(-1), energy.view(-1))
        return energy


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout=0.1, device='cpu'):
        super(LuongAttnDecoderRNN, self).__init__()

        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.device = device
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size, self.device)

    def forward(self, input_seq, last_hidden, encoder_outputs):
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)
        # 因为默认 GRU 是 batch_first False 的
        embedded = embedded.unsqueeze(0)

        rnn_output, hidden = self.gru(embedded, last_hidden)

        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))

        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))

        output = self.out(concat_output)

        return output, hidden, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, config, data_feature):
        super(Seq2Seq, self).__init__()
        self.pad_token = data_feature['loc_num']
        self.device = config['device']
        self.n_layers = config['n_layers']
        self.encoder = EncoderRNN(data_feature['loc_num'] + 1, config['hidden_size'], config['n_layers'],
                                  dropout=config['dropout'])
        self.decoder = LuongAttnDecoderRNN(config['attn_model'], config['hidden_size'], data_feature['loc_num'] + 1,
                                           config['n_layers'], dropout=config['dropout'], device=config['device'])

    def forward(self, batch):
        input_seq = batch['input_seq']
        seq_length = batch['seq_length']
        target_seq = batch['target_seq']
        encoder_outputs, encoder_hidden = self.encoder(input_seq, seq_length)
        batch_size = input_seq.size(0)
        # 使用 road pad 作为 decoder 的初始输入
        decoder_input = torch.LongTensor([self.pad_token] * batch_size).to(self.device)
        decoder_hidden = encoder_hidden[:self.n_layers]
        max_length = max(seq_length)
        all_decoder_output = torch.zeros((max_length, input_seq.shape[0], self.decoder.output_size)).to(self.device)
        target_seq = target_seq.transpose(0, 1)  # 使得 seq_len 在第一维度
        for t in range(max_length):
            decoder_output, decoder_hidden, decoder_attn = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            all_decoder_output[t] = decoder_output
            decoder_input = target_seq[t]
        return all_decoder_output



