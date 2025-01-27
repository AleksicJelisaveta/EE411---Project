import torch

class ElasticWeightConsolidation:
    def __init__(self, model, importance=1000):
        self.model = model
        self.importance = importance
        self.params = {n: p.clone().detach() for n, p in model.named_parameters()}
        self.fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}

    def update_fisher(self, data_loader, criterion):
        self.model.eval()
        for inputs, targets in data_loader:
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            for n, p in self.model.named_parameters():
                self.fisher[n] += p.grad ** 2 / len(data_loader)

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self.fisher[n] * (p - self.params[n]) ** 2
            loss += _loss.sum()
        return self.importance * loss
