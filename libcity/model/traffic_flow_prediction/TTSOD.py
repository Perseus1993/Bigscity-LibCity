import math
import pickle

import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
# import matplotlib.pyplot as plt

from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss


class LearnedPositionalEncoding(nn.Embedding):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # [5625, 1, 64]
        weight = self.weight.data.unsqueeze(1)
        x = x + weight[:x.size(0), :]
        return self.dropout(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        max_len = 5625
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # pe = torch.zeros(max_len, d_model)
        pe = torch.zeros(max_len, 1, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0, 0::2] = torch.sin(position * div_term)  # 修改此处
        pe[:, 0, 1::2] = torch.cos(position * div_term)  # 修改此处

        # pe[:, 0::2] = torch.sin(position * div_term)
        # pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print("x shape", x.shape)
        # print("pe shape", self.pe.shape)
        # x = x + self.pe[:x.size(0), :]
        x = x + self.pe[:x.size(0), :, :]
        return self.dropout(x)


class TemproalEmbedding(nn.Module):
    def __init__(self, in_dim, layers=1, dropout=.1):
        super(TemproalEmbedding, self).__init__()
        self.rnn = nn.LSTM(input_size=in_dim, hidden_size=in_dim, num_layers=layers, dropout=dropout)

    def forward(self, input):
        # (batch,hid_dim,sensor,len)
        ori_shape = input.shape
        # x = (len,batch,sensor,hid_dim)
        x = input.permute(3, 0, 2, 1)
        # x = (len,batch*sensor,hid_dim)
        x = x.reshape(ori_shape[3], ori_shape[0] * ori_shape[2], ori_shape[1])
        x, _ = self.rnn(x)
        # (len, batch, sensor, hid_dim)
        x = x.reshape(ori_shape[3], ori_shape[0], ori_shape[2], ori_shape[1])
        # (batch, hid_dim, sensor, len)
        x = x.permute(1, 3, 2, 0)
        return x


class TrafficTransformer(nn.Module):
    def __init__(self, in_dim, layers=1, dropout=.1, heads=8):
        super().__init__()
        self.heads = heads
        self.pos = PositionalEncoding(in_dim, dropout=dropout)
        self.lpos = LearnedPositionalEncoding(in_dim, dropout=dropout)
        self.trans = nn.Transformer(in_dim, heads, layers, layers, in_dim * 4, dropout=dropout)

    def forward(self, input, mask):
        x = input.permute(1, 0, 2)
        # print("before pos",x.shape)
        x = self.pos(x)
        # x = self.lpos(x)
        x = self.trans(x, x, tgt_mask=mask)
        return x.permute(1, 0, 2)

    def _gen_mask(self, input):
        l = input.shape[1]
        mask = torch.eye(l)
        mask = mask.bool()
        return mask


class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class TTSOD(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.device = config.get('device', torch.device('cpu'))
        self.batch_size = config.get('batch_size')
        self.output_dim = config.get('output_dim')

        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)
        self._scaler = self.data_feature.get('scaler')
        self.height = data_feature.get('len_row', 15)
        self.width = data_feature.get('len_column', 5)

        self.in_dim = config.get('in_dim', 1)
        self.hid_dim = config.get('hid_dim', 64)
        self.layers = config.get('layers', 1)
        self.dropout = config.get('dropout', 0.1)
        self.out_dim = config.get('output_dim', 1)
        self.supports = self._build_supports()
        self.mask = self._build_masks()

        self.start_conv = nn.Conv2d(in_channels=self.in_dim,
                                    out_channels=self.hid_dim,
                                    kernel_size=(1, 1))

        self.start_embedding = TemproalEmbedding(self.hid_dim, layers=3, dropout=self.dropout)
        self.end_conv = nn.Linear(self.hid_dim, self.out_dim)
        self.network = TrafficTransformer(in_dim=self.hid_dim, layers=self.layers, dropout=self.dropout)

    def _build_supports(self):
        sensor_ids, sensor_id_to_ind, adj_mx = self._load_adj('libcity/data/sensor_graph/adj_mx.pkl')
        return [torch.tensor(i).to(self.device) for i in adj_mx]

    def _load_adj(self, pkl_filename):
        def load_pickle(pickle_file):
            try:
                with open(pickle_file, 'rb') as f:
                    pickle_data = pickle.load(f)
            except UnicodeDecodeError as e:
                with open(pickle_file, 'rb') as f:
                    pickle_data = pickle.load(f, encoding='latin1')
            except Exception as e:
                print('Unable to load data ', pickle_file, ':', e)
                raise
            return pickle_data

        def asym_adj(adj):
            adj = sp.coo_matrix(adj)
            rowsum = np.array(adj.sum(1)).flatten()
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)
            return d_mat.dot(adj).astype(np.float32).todense()

        sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
        return sensor_ids, sensor_id_to_ind, adj

    def _build_masks(self):
        mask0 = self.supports[0].detach()
        mask1 = self.supports[1].detach()
        mask = mask0 + mask1
        out = 0
        for i in range(1, 7):
            out += mask ** i
        return out == 0

    def forward(self, batch):
        x = batch['X']
        b_s = x.shape[0]
        # print("====",x.shape) [16, 12, 207, 1]
        # #打印x的第一个batch的207个传感器画出来

        x = x.transpose(1, 3)
        # print("x shape 1",x.shape) #x shape 1 torch.Size([16, 5, 15, 12, 15, 5, 1])
        # [32, 5, 15, 12, 15, 5, 1]转换为[32, 1, 75*75,12]
        x = x.reshape(b_s, 1, 75 * 75, 12)
        # print("x shape 2",x.shape)
        x = self.start_conv(x)
        # print("after start_conv", x.shape)
        x = self.start_embedding(x)[..., -1]
        # print("after start_embedding", x.shape)
        x = x.transpose(1, 2)
        # x = self.network(x, self.mask)
        # x = self.network(x, torch.ones((5625, 5625)).to(x.device))
        x = self.network(x, torch.ones((769, 769)).to(x.device))

        x = self.end_conv(x)
        # print("x shape 3",x.shape)
        return x.transpose(1, 2).unsqueeze(-1)

    def predict(self, batch):
        y_predicted = self.forward(batch)
        # scaler = StandardScaler(mean=batch['X'][..., 0].mean(), std=batch['X'][..., 0].std())
        # y_predicted = scaler.inverse_transform(y_predicted)
        y_predicted = y_predicted.reshape(y_predicted.shape[0], 12, 15, 5, 15, 5, 1)
        return y_predicted

    def calculate_loss(self, batch):
        y_true = batch['y']
        # y_true = y_true.reshape(y_true.shape[0],  12, 5625, 1)
        # print("y_true shape",y_true.shape)
        y_predicted = self.predict(batch)

        # print("y_predicted shape",y_predicted.shape)

        def masked_mae(preds, labels, null_val=np.nan):
            if np.isnan(null_val):
                mask = ~torch.isnan(labels)
            else:
                mask = (labels != null_val)
            mask = mask.float()
            mask /= torch.mean(mask)
            mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
            loss = torch.abs(preds - labels)
            loss = loss * mask
            loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
            return torch.mean(loss)

        # return loss.masked_mse_torch(y_predicted, y_true, 0)
        return masked_mae(y_predicted, y_true, 0)
