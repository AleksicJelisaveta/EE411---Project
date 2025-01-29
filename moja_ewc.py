
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import copy
import torch.nn.functional as F


class EWC:
    def __init__(self, model):
        self.model = model
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in self.params.items():
            precision_matrices[n] = p.clone().detach().fill_(0)
        return precision_matrices

    def compute_fisher(self, data_loader):
        self.model.eval()
        for inputs, labels in data_loader:
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = F.nll_loss(outputs, labels)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    self._precision_matrices[n] += p.grad ** 2 / len(data_loader)

    def update_params(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self._means[n] = p.clone().detach()

    def compute_ewc_loss(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
                loss += _loss.sum()
        return loss