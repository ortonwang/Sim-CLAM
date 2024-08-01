# -*- coding: utf-8 -*-

import os
import sys
import time
import numpy as np
import torch
import random
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import Counter

sys.path.append('..')





class ElrLoss(nn.Module):
    def __init__(self, num_examp, num_classes, epoch):
        super(ElrLoss, self).__init__()
        self.q = 0
        self.pred_hist = torch.zeros(num_examp, num_classes)
        self.lamb = np.linspace(1, min(10, epoch), epoch)
        self.beta = self.lamb[::-1] / 10
        # self.weight = weight.cuda()
        self.weight_t = torch.ones(num_classes)

    def update_weight(self):
        t = torch.argmax(self.pred_hist, dim=1)
        label_count = dict(Counter(t.numpy()))
        weight = torch.ones(2)
        for i in range(2):
            weight[i] = mean_class_num / label_count[i] if i in label_count else 0
        weight[weight == 0] = max(weight)
        self.weight_t = weight

    def forward(self, output, y_labeled, epoch):
        t_batch = torch.argmax(self.q, dim=1).cpu()
        weight_batch = self.weight_t[t_batch]
        y_pred = F.softmax(output, dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)
        ce_loss = F.cross_entropy(output, y_labeled)
        reg = ((1 - (self.q * y_pred).sum(dim=1)).log())
        reg = (reg * weight_batch.cuda()).sum() / weight_batch.sum()
        final_loss = ce_loss + self.lamb[epoch] * reg
        return final_loss

    def update_hist(self, epoch, output, index=None):
        y_pred_ = F.softmax(output, dim=1).cpu()
        self.pred_hist[index] = self.beta[epoch] * self.pred_hist[index]\
                                + (1 - self.beta[epoch]) * y_pred_ / (y_pred_).sum(dim=1, keepdim=True)
        self.q = self.pred_hist[index].cuda()



def get_label(net, test_loader):
    net.eval()
    labels = torch.zeros(len(test_loader.dataset), dtype=torch.long).cuda()

    with torch.no_grad():
        for _, (inputs, _, index) in enumerate(test_loader):
            inputs = inputs.cuda()
            outputs = net(inputs)
            labels[index] = torch.max(outputs, dim=-1)[1]
    return np.array(labels.cpu())


def train_elr_adapt(train_loader, model, optimizer, epoch):
    print('\nEpoch: %d' % epoch)

    for batch_idx, (inputs, targets, indexs) in enumerate(train_loader):
        model.train()
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)

        elr.update_hist(epoch, outputs.data.detach(), indexs.numpy().tolist())
        loss = elr(outputs, targets, epoch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# loss


num_examp = 200

mean_class_num = int(num_examp/2)

elr = ElrLoss(num_examp=num_examp, weight=None, num_classes=2, epoch=200)






