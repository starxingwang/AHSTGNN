# @Time     : 01. 07, 2022 16:57:
# @Author   : Xing Wang, Kexin Yang
# @FileName : util.py
# @Version  : 1.0
# @IDE      : PyCharm
# @Github   : https://github.com/starxingwang/AHSTGNN

import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg


class DataLoader(object):
    def __init__(self, xs1, xs2, xs3, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs1) % batch_size)) % batch_size
            x1_padding = np.repeat(xs1[-1:], num_padding, axis=0)
            x2_padding = np.repeat(xs2[-1:], num_padding, axis=0)
            x3_padding = np.repeat(xs3[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs1 = np.concatenate([xs1, x1_padding], axis=0)
            xs2 = np.concatenate([xs2, x2_padding], axis=0)
            xs3 = np.concatenate([xs3, x3_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs1)
        self.num_batch = int(self.size // self.batch_size)
        self.xs1 = xs1
        self.xs2 = xs2
        self.xs3 = xs3
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs1, xs2, xs3, ys = self.xs1[permutation], self.xs2[permutation], self.xs3[permutation], self.ys[permutation]
        self.xs1 = xs1
        self.xs2 = xs2
        self.xs3 = xs3
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x1_i = self.xs1[start_ind: end_ind, ...]
                x2_i = self.xs2[start_ind: end_ind, ...]
                x3_i = self.xs3[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x1_i, x2_i, x3_i, y_i)
                self.current_ind += 1

        return _wrapper()


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


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def asym_adj(adj):
    # random walk
    adj = sp.coo_matrix(adj)
    # rowsum(A)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()


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


def load_adj(pkl_filename, adjtype):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return sensor_ids, sensor_id_to_ind, adj


def load_dataset(dataset_dir, batch_size):
    data = {}
    # for category in ['train', 'val', 'test']:
    for category in ['train', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'), allow_pickle=True)
        data['recent_' + category] = cat_data['hour']
        data['day_' + category] = cat_data['day']
        data['week_' + category] = cat_data['week']
        data['target_' + category] = cat_data['target']
    scaler_recent = StandardScaler(mean=data['recent_train'][..., 0].mean(), std=data['recent_train'][..., 0].std())
    scaler_day = StandardScaler(mean=data['day_train'][..., 0].mean(), std=data['day_train'][..., 0].std())
    scaler_week = StandardScaler(mean=data['week_train'][..., 0].mean(), std=data['week_train'][..., 0].std())
    # Data format
    # for category in ['train', 'val', 'test']:
    for category in ['train', 'test']:
        data['recent_' + category][..., 0] = scaler_recent.transform(data['recent_' + category][..., 0])
        data['day_' + category][..., 0] = scaler_day.transform(data['day_' + category][..., 0])
        data['week_' + category][..., 0] = scaler_week.transform(data['week_' + category][..., 0])
    data['train_loader'] = DataLoader(data['recent_train'], data['day_train'], data['week_train'], data['target_train'],
                                      batch_size)
    # data['val_loader'] = DataLoader(data['recent_val'], data['day_val'], data['week_val'], data['target_val'],
                                    # batch_size)
    data['test_loader'] = DataLoader(data['recent_test'], data['day_test'], data['week_test'], data['target_test'],
                                     batch_size)
    data['scaler'] = scaler_recent
    return data


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


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


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse


def print_model_parameters(model, only_num=True):
    print('*****************Model Parameter*****************')
    if not only_num:
        for name, param in model.named_parameters():
            print(name, param.shape, param.requires_grad)
    total_num = sum([param.nelement() for param in model.parameters()])
    print('Total params num: {}'.format(total_num))
    print('*****************Finish Parameter****************')
