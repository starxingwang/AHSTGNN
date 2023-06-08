# @Time     : 01. 07, 2022 16:57:
# @Author   : Xing Wang, Kexin Yang
# @FileName : engine.py
# @Version  : 1.0
# @IDE      : PyCharm
# @Github   : https://github.com/starxingwang/AHSTGNN

import torch.optim as optim
from model import *
import util
from util import print_model_parameters

class trainer():
    def __init__(self, scaler, in_dim, seq_length, num_nodes, nhid, dropout, lrate, wdecay, device, supports, gcn_bool,
                 addaptadj, aptinit):
        self.model = gwnet(device, num_nodes, dropout, supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj,
                           aptinit=aptinit, in_dim=in_dim, out_dim=seq_length, residual_channels=nhid,
                           dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16)
        print_model_parameters(self.model, only_num=False)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5

    def train(self, input1, input2, input3, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        input1 = nn.functional.pad(input1, (1, 0, 0, 0))
        input2 = nn.functional.pad(input2, (1, 0, 0, 0))
        input3 = nn.functional.pad(input3, (1, 0, 0, 0))
        output = self.model([input1, input2, input3])
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=1)
        loss = self.loss(output, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = util.masked_mape(output, real, 0.0).item()
        rmse = util.masked_rmse(output, real, 0.0).item()
        return loss.item(), mape, rmse

    def eval(self, input1, input2, input3, real_val):
        self.model.eval()
        input1 = nn.functional.pad(input1, (1, 0, 0, 0))
        input2 = nn.functional.pad(input2, (1, 0, 0, 0))
        input3 = nn.functional.pad(input3, (1, 0, 0, 0))
        output = self.model([input1, input2, input3])
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=1)
        loss = self.loss(output, real, 0.0)
        mape = util.masked_mape(output, real, 0.0).item()
        rmse = util.masked_rmse(output, real, 0.0).item()
        return loss.item(), mape, rmse
