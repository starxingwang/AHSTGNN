# @Time     : 01. 07, 2022 16:57:
# @Author   : Xing Wang, Kexin Yang
# @FileName : test.py
# @Version  : 1.0
# @IDE      : PyCharm
# @Github   : https://github.com/starxingwang/AHSTGNN
import util
import argparse
from model import *
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:1', help='')
parser.add_argument('--data', type=str, default='data/25_10_0_period', help='data path')
parser.add_argument('--adjdata', type=str, default='data/sensor_graph/nj_0_adj_mat.pkl', help='adj data path')
parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type')
parser.add_argument('--gcn_bool', action='store_true', help='whether to add graph convolution layer')
parser.add_argument('--aptonly', action='store_true', help='whether only adaptive adj')
parser.add_argument('--addaptadj', action='store_true', help='whether add adaptive adj')
parser.add_argument('--randomadj', action='store_true', help='whether random initialize adaptive adj')
parser.add_argument('--seq_length', type=int, default=12, help='')
parser.add_argument('--nhid', type=int, default=64, help='')
parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
parser.add_argument('--num_nodes', type=int, default=1036, help='number of nodes')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--checkpoint', type=str, help='')

args = parser.parse_args()


def main():
    device = torch.device(args.device)

    _, _, adj_mx = util.load_adj(args.adjdata, args.adjtype)
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None

    model = gwnet(device, args.num_nodes, args.dropout, supports=supports, gcn_bool=args.gcn_bool,
                  addaptadj=args.addaptadj, aptinit=adjinit)
    model.to(device)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()

    print('model load successfully')

    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    outputs = []
    target = dataloader['target_test'].astype(float)
    realy = torch.Tensor(target).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    for iter, (x1, x2, x3, y) in enumerate(dataloader['test_loader'].get_iterator()):
        x1 = x1.astype(float)
        x2 = x2.astype(float)
        x3 = x3.astype(float)
        y = y.astype(float)
        testx1 = torch.Tensor(x1).to(device)
        testx1 = testx1.transpose(1, 3)
        testx2 = torch.Tensor(x2).to(device)
        testx2 = testx2.transpose(1, 3)
        testx3 = torch.Tensor(x3).to(device)
        testx3 = testx3.transpose(1, 3)
        testx1 = nn.functional.pad(testx1, (1, 0, 0, 0))
        testx2 = nn.functional.pad(testx2, (1, 0, 0, 0))
        testx3 = nn.functional.pad(testx3, (1, 0, 0, 0))

        with torch.no_grad():
            preds = model([testx1, testx2, testx3]).transpose(1, 3)
            print(preds.shape)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]
    print('yhat', yhat.shape)

    amae = []
    amape = []
    armse = []
    for i in range(12):
        pred = yhat[:, :, i]
        real = realy[:, :, i]
        metrics = util.metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))


if __name__ == "__main__":
    main()
