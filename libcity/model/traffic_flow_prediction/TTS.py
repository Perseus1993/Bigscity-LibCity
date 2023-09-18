import math
import pickle

import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
# import matplotlib.pyplot as plt

from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss

import pandas as pd

class LearnedPositionalEncoding(nn.Embedding):
    def __init__(self, d_model, dropout=0.1, max_len=800):
        super().__init__(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        weight = self.weight.data.unsqueeze(1)
        x = x + weight[:x.size(0), :]
        return self.dropout(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1355):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
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


class TTS(AbstractTrafficStateModel):
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

        self.in_dim = config.get('in_dim', 3)
        self.hid_dim = config.get('hid_dim', 64)
        self.layers = config.get('layers', 1)
        self.dropout = config.get('dropout', 0.1)
        self.out_dim = config.get('output_dim', 3)
        self.supports = self._build_supports()
        self.mask = self._build_masks()

        self.start_conv = nn.Conv2d(in_channels=self.in_dim,
                                    out_channels=self.hid_dim,
                                    kernel_size=(1, 1))

        self.start_embedding = TemproalEmbedding(self.hid_dim, layers=3, dropout=self.dropout)
        self.end_conv = nn.Linear(self.hid_dim, self.out_dim * 3)
        self.network = TrafficTransformer(in_dim=self.hid_dim, layers=self.layers, dropout=self.dropout)
        with open(r'libcity/data/adj_mx_1355.pkl', 'rb') as f:
            self.adj_test = pickle.load(f)

        with open(r'libcity/exp/tot_embedding_dict.pkl', 'rb') as f:
            self.tot_embedding_dict = pickle.load(f)
        print("read embedding_dict success")


        # 读取文件
        df = pd.read_csv('raw_data/CT/CT.dyna')

        # 获取唯一的 'geo_id'
        unique_geo_ids = df['entity_id'].unique()
        unique_list = unique_geo_ids.tolist()
        print("unique_list", len(unique_list))
        print("unique_list", unique_list)
        embeddings_list = [self.tot_embedding_dict[geo_id] for geo_id in unique_list]
        self.embeddings_tensor = torch.stack(embeddings_list)





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
        print("====",x.shape)
        # #打印x的第一个batch的207个传感器画出来
        # print("1---x shape",x.shape)
        x = x.transpose(1, 3)
        x = self.start_conv(x)
        x = self.start_embedding(x)[..., -1]
        x = x.transpose(1, 2)
        # x = self.network(x, self.mask)
        # 读取E:\develop\gnn_cargo\embedding\adj_mx_test.pkl

        mask_test = torch.ones((1355, 1355)).to(x.device)
        # mask_test 在 adj_test   =  1的地方为0
        # mask_test[self.adj_test == 1] = 0

        x = self.network(x, mask_test)
        # x = self.network(x, torch.ones((769, 769)).to(x.device))
        # print("2---x shape", x.shape)

        x = self.end_conv(x)
        # print("3---x shape", x.shape)
        # [32, 1355, 36]转成[32, 12, 1355, 3]
        x = x.reshape(x.shape[0], self.output_window, x.shape[1], -1)
        # print("4---x shape", x.shape)

        return x

    def predict(self, batch):
        y_predicted = self.forward(batch)
        # scaler = StandardScaler(mean=batch['X'][..., 0].mean(), std=batch['X'][..., 0].std())
        # y_predicted = scaler.inverse_transform(y_predicted)
        return y_predicted

    def calculate_loss(self, batch):
        y_true = batch['y']
        # y_true = y_true.transpose(1, 3)
        y_predicted = self.predict(batch)

        # print("===================================")

        # print("y_true", y_true[0, :12, 450])
        # print("y_predicted", y_predicted[0, :12, 450])
        # print("===================================")

        # y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])

        # y_true = torch.unsqueeze(y_true, dim=1)
        # print("y_predicted_before", y_predicted[0, :, 0])

        # print("scaler", self._scaler)
        #
        # print("y_true", y_true.shape)
        # print("y_predicted", y_predicted.shape)
        #
        # print("y_true", y_true[0, :, 0])
        # print("y_predicted", y_predicted[0, :, 0])

        def masked_mae(preds, labels, null_val=np.nan):
            if np.isnan(null_val):
                mask = ~torch.isnan(labels)
            else:
                mask = (labels != null_val)
            mask = mask.float()
            mask /= torch.mean((mask))
            mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
            loss = torch.abs(preds - labels)
            loss = loss * mask
            loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
            return torch.mean(loss)

        return loss.masked_mse_torch(y_predicted, y_true, 0)
        # return masked_mae(y_predicted, y_true, 0)

