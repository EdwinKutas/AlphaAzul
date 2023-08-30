import torch
from torch import nn
import torch.nn.functional as F

import os
import sys
import time

import numpy as np
from tqdm import tqdm


import torch.optim as optim
from utils import *


sys.path.append('../../')

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,
})


class AzulNet(nn.Module):

    def __init__(self, hyper_params):
        super(AzulNet, self).__init__()
        self.tiles_linear_1 = nn.Linear(31, hyper_params[0])
        self.board_linear_1 = nn.Linear(25, hyper_params[1])
        self.rows_linear_1 = nn.Linear(11, hyper_params[2])
        dim = hyper_params[0] * hyper_params[1]**2 * hyper_params[2]**2 * 6
        self.final_linear_head = nn.Linear(dim, 180)
        self.final_linear_tail = nn.Linear(dim, 1)
        self.bn_1 = nn.BatchNorm1d(dim)


    def forward(self, x):
        """
        This is virtually a Linear mapping.
        """
        tiles = x[0:31]
        board_1 = x[31:56]
        rows_1 = x[56:67]
        points_1 = x[67:68]/100
        board_2 = x[68:93]
        rows_2 = x[93:104]
        points_2 = x[104:105]/100
        tiles_in_play = x[105:110]

        tiles = self.tiles_linear_1(tiles)
        board_1 = self.board_linear_1(board_1)
        board_2 = self.board_linear_1(board_2)
        rows_1 = self.rows_linear_1(rows_1)
        rows_2 = self.rows_linear_1(rows_2)

        inputs = [tiles, board_1, rows_1, points_1, board_2, rows_2, points_2, tiles_in_play]
        for i in range(0, len(inputs)):
            inputs[i] = torch.cat((inputs[i], torch.tensor([1])), 0)

        out = inputs[0]
        for i in range(1, len(inputs)):
            out = torch.outer(out, inputs[i])
            out = out.flatten()
        out = out.unsqueeze(0)
        out = self.bn_1(out)
        return F.log_softmax(self.final_linear_head(out)), F.tanh(self.final_linear_tail(out))


class NNetWrapper():

    def __init__(self, args):
        self.nnet = AzulNet(args)
        self.action_size = 180

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters())

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / args.batch_size)

            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # compute output
                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return pi, v

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder=r'C:\Users\user\PycharmProjects\Azul\Neural nets\Final Round', filename='round_1.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])


