# @Time     : 01. 07, 2022 16:57:
# @Author   : Xing Wang, Kexin Yang
# @FileName : train.py
# @Version  : 1.0
# @IDE      : PyCharm
# @Github   : https://github.com/starxingwang/AHSTGNN
import argparse
import time
import util
from engine import trainer
from model import *
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--data', type=str, default='data/Milan', help='data path')
parser.add_argument('--adjdata', type=str, default='data/sensor_graph/adj_mat.pkl', help='adj data path')
parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type')
parser.add_argument('--gcn_bool', action='store_true', help='whether to add graph convolution layer')
parser.add_argument('--aptonly', action='store_true', help='whether only adaptive adj')
parser.add_argument('--addaptadj', action='store_true', help='whether add adaptive adj')
parser.add_argument('--randomadj', action='store_true', help='whether random initialize adaptive adj')
parser.add_argument('--seq_length', type=int, default=6, help='')
parser.add_argument('--nhid', type=int, default=32, help='')
parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
parser.add_argument('--num_nodes', type=int, default=900, help='number of nodes')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--epochs', type=int, default=2, help='')
parser.add_argument('--print_every', type=int, default=50, help='')
parser.add_argument('--save', type=str, default='./garage/milan', help='save path')
parser.add_argument('--expid', type=int, default=1, help='experiment id')

args = parser.parse_args()


def main():
    # load data
    device = torch.device(args.device)
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata, args.adjtype)
    dataloader = util.load_dataset(args.data, args.batch_size)
    scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx]

    print(args)

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None

    engine = trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                     args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                     adjinit)

    print("start training...", flush=True)
    his_loss = []
    val_time = []
    train_time = []
    # training
    for i in range(1, args.epochs + 1):
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x1, x2, x3, y) in enumerate(dataloader['train_loader'].get_iterator()):
            x1 = x1.astype(float)
            x2 = x2.astype(float)
            x3 = x3.astype(float)
            y = y.astype(float)
            trainx1 = torch.Tensor(x1).to(device)
            trainx1 = trainx1.transpose(1, 3)
            trainx2 = torch.Tensor(x2).to(device)
            trainx2 = trainx2.transpose(1, 3)
            trainx3 = torch.Tensor(x3).to(device)
            trainx3 = trainx3.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            metrics = engine.train(trainx1, trainx2, trainx3, trainy[:, 0, :, :])
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)
        t2 = time.time()
        train_time.append(t2 - t1)

        # validation
        # Since the milan dataset doesn't have the validation set, we commented out this part
        # valid_loss = []
        # valid_mape = []
        # valid_rmse = []
        #
        # s1 = time.time()
        # for iter, (x1, x2, x3, y) in enumerate(dataloader['val_loader'].get_iterator()):
        #     x1 = x1.astype(float)
        #     x2 = x2.astype(float)
        #     x3 = x3.astype(float)
        #     y = y.astype(float)
        #     valx1 = torch.Tensor(x1).to(device)
        #     valx1 = valx1.transpose(1, 3)
        #     valx2 = torch.Tensor(x2).to(device)
        #     valx2 = valx2.transpose(1, 3)
        #     valx3 = torch.Tensor(x3).to(device)
        #     valx3 = valx3.transpose(1, 3)
        #     valy = torch.Tensor(y).to(device)
        #     valy = valy.transpose(1, 3)
        #     metrics = engine.eval(valx1, valx2, valx3, valy[:, 0, :, :])
        #     valid_loss.append(metrics[0])
        #     valid_mape.append(metrics[1])
        #     valid_rmse.append(metrics[2])
        # s2 = time.time()
        # log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        # print(log.format(i, (s2 - s1)))
        # val_time.append(s2 - s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)
        his_loss.append(mtrain_loss)
        
        # mvalid_loss = np.mean(valid_loss)
        # mvalid_mape = np.mean(valid_mape)
        # mvalid_rmse = np.mean(valid_rmse)
        # his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, (t2 - t1)), flush=True)
        torch.save(engine.model.state_dict(),
                   args.save + "_epoch_" + str(i) + "_" + str(round(mtrain_loss, 2)) + ".pth")
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))

        # log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        # print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),
        #       flush=True)
        # torch.save(engine.model.state_dict(),
        #            args.save + "_epoch_" + str(i) + "_" + str(round(mvalid_loss, 2)) + ".pth")

    # print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    # print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))
    

    # testing
    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(
        torch.load(args.save + "_epoch_" + str(bestid + 1) + "_" + str(round(his_loss[bestid], 2)) + ".pth"))
    
    outputs = []
    target = dataloader['target_test'].astype(float)
    realy = torch.Tensor(target).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    engine.model.eval()
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
            preds = engine.model([testx1, testx2, testx3]).transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid], 4)))

    amae = []
    amape = []
    armse = []
    for i in range(6):
        pred = yhat[:, :, i]
        real = realy[:, :, i]
        metrics = util.metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over 6 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))
    torch.save(engine.model.state_dict(),
               args.save + "_exp" + str(args.expid) + "_best_" + str(round(his_loss[bestid], 2)) + ".pth")


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))
