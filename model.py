# @Time     : 01. 07, 2022 16:57:
# @Author   : Xing Wang, Kexin Yang
# @FileName : model.py
# @Version  : 1.0
# @IDE      : PyCharm
# @Github   : https://github.com/starxingwang/AHSTGNN

import torch
import torch.nn as nn
from torch.nn import ModuleList
import torch.nn.functional as F


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', (x, A))  # Matrix multiplication
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class ComputeAttentionScore(nn.Module):
    def __init__(self):
        super(ComputeAttentionScore, self).__init__()

    def forward(self, x, node_vec):
        n_q = node_vec.unsqueeze(dim=-1)
        x_t_a = torch.einsum('btnd,ndl->btnl', (x, n_q)).contiguous()
        return x_t_a


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        # attention——> [B, N, N]
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.transpose(2, 3)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, n_in, n_out, dropout, alpha, nheads, order=1):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.nheads = nheads
        self.order = order

        self.attentions = [GraphAttentionLayer(n_in, n_out, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        for k in range(2, self.order + 1):
            self.attentions_2 = ModuleList(
                [GraphAttentionLayer(n_in, n_out, dropout=dropout, alpha=alpha, concat=True) for _ in
                 range(nheads)])

        self.out_att = GraphAttentionLayer(n_out * nheads * order, n_out, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        for k in range(2, self.order + 1):
            x2 = torch.cat([att(x, adj) for att in self.attentions_2], dim=-1)
            x = torch.cat([x, x2], dim=-1)
        x = F.elu(self.out_att(x, adj))
        return x


class Gate(nn.Module):
    def __init__(self, n_out):
        """Dense version of GAT."""
        super(Gate, self).__init__()
        self.n_out = n_out

        self.W_z = nn.Parameter(torch.empty(size=(2 * n_out, n_out)))
        nn.init.xavier_uniform_(self.W_z.data, gain=1.414)
        self.b = nn.Parameter(torch.empty(size=(1, n_out)))
        nn.init.xavier_uniform_(self.b.data, gain=1.414)

    def forward(self, x, h):
        x_h = torch.cat((x, h), dim=-1)  # concat x and h_(t-1)
        Wh = torch.matmul(x_h, self.W_z)
        gate = torch.sigmoid(Wh + self.b)
        one_vec = torch.ones_like(gate)
        z = gate * x + (one_vec - gate) * h
        return z


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class gated_TCN(nn.Module):
    def __init__(self, residual_channels=32, dilation_channels=32, kernel_size=2, layers=2):
        super(gated_TCN, self).__init__()
        self.layers = layers
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()

        new_dilation = 1
        for _ in range(layers):
            # dilated convolutions
            self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                               out_channels=dilation_channels,
                                               kernel_size=(1, kernel_size), dilation=new_dilation))

            self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                             out_channels=dilation_channels,
                                             kernel_size=(1, kernel_size), dilation=new_dilation))

            new_dilation *= 2

    def forward(self, input):
        x = input
        for i in range(self.layers):
            residual = x
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate
        return x


class gwnet(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True,
                 aptinit=None,
                 in_dim=1, out_dim=6, residual_channels=32, dilation_channels=32,
                 skip_channels=256, end_channels=512,
                 kernel_size=2, blocks=2, layers=2, apt_size=10,
                 alpha=0.2, nheads=4):
        super(gwnet, self).__init__()
        self.ComputeAttentionScore = ComputeAttentionScore()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.tcn1 = nn.ModuleList()
        self.tcn2 = nn.ModuleList()
        self.tcn3 = nn.ModuleList()

        self.gconv = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        # Add four query W
        self.w_t = nn.ModuleList()
        self.w_ls = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))

        self.mlp = nn.Conv2d(in_channels=dilation_channels * 3, out_channels=dilation_channels, kernel_size=(1, 1),
                             padding=(0, 0), stride=(1, 1), bias=True)

        self.supports = supports

        receptive_field = 1
        for _ in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for _ in range(layers):
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, apt_size).to(device), requires_grad=True).to(
                    device)  # (num_nodes,10)
                self.nodevec2 = nn.Parameter(torch.randn(apt_size, num_nodes).to(device), requires_grad=True).to(
                    device)  # (10,num_nodes)
                self.supports_len += 1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1

        for b in range(blocks):
            self.tcn1.append(gated_TCN(residual_channels, dilation_channels, kernel_size, layers))
            self.tcn2.append(gated_TCN(residual_channels, dilation_channels, kernel_size, layers))
            self.tcn3.append(gated_TCN(residual_channels, dilation_channels, kernel_size, layers))
            self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                             out_channels=skip_channels,
                                             kernel_size=(1, 1)))
            self.bn.append(nn.BatchNorm2d(residual_channels))

            if self.gcn_bool:
                self.supports_len = 1
                self.gconv.append(gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len))

            # Add four query W
            self.w_t.append(nn.Conv2d(in_channels=apt_size,
                                      out_channels=residual_channels,
                                      kernel_size=(1, 1)))
            self.w_ls.append(nn.Conv2d(in_channels=apt_size,
                                       out_channels=residual_channels,
                                       kernel_size=(1, 1)))

        # Add GAT
        depth = list(range(blocks * layers))  # blocks=4, layers=2
        self.GATs = ModuleList([GAT(residual_channels, residual_channels, dropout, alpha, nheads)
                                for _ in depth])
        self.Gates = ModuleList([Gate(residual_channels) for _ in depth])

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field

    def forward(self, input):
        recent, daily, weekly = input
        x1 = self.start_conv(recent)
        x2 = self.start_conv(daily)
        x3 = self.start_conv(weekly)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            adp = torch.eye(adp.shape[0]).to(adp.device) + adp
            new_supports = [adp]

        # WaveNet layers
        for i in range(self.blocks):
            out = []
            x1 = self.tcn1[i](x1)
            x2 = self.tcn2[i](x2)
            x3 = self.tcn3[i](x3)
            out.append(x1)
            out.append(x2)
            out.append(x3)

            x = torch.cat(out, dim=1)
            x = self.mlp(x)

            x_t = x.transpose(1, 3)

            # Add Time attention
            # compute Time attention score
            n_q_t = self.w_t[i](self.nodevec1.unsqueeze(dim=-1).unsqueeze(dim=-1)).squeeze()
            x_t_a = self.ComputeAttentionScore(x_t, n_q_t)
            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x, self.supports)
            else:
                x = self.residual_convs[i](x)

            x = self.bn[i](x)

            # Long/Static Spatial feature x_ls
            x_ls = x.transpose(1, 3)

            # Add GAT
            x_ds = self.GATs[i](x_t, self.supports[0])
            x_ls = self.Gates[i](x_ls, x_ds)

            # Add Long/Static Spatial attention
            # compute Long/Static Spatial attention score
            n_q_ls = self.w_ls[i](self.nodevec1.unsqueeze(dim=-1).unsqueeze(dim=-1)).squeeze()
            x_ls_a = self.ComputeAttentionScore(x_ls, n_q_ls)

            # node-level adaptation tendencies
            x_a = torch.cat((x_t_a, x_ls_a), -1)
            x_att = F.softmax(x_a, dim=-1)

            # Add Time, Long/Static attention
            x = x_att[:, :, :, 0].unsqueeze(dim=-1) * x_t + x_att[:, :, :, 1].unsqueeze(dim=-1) * x_ls
            x = x.transpose(1, 3)

            # parametrized skip connection
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = s + skip

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x
