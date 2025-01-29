import torch

class ElasticWeightConsolidation:
    def __init__(self, model, importance=1000):
        self.model = model
        self.importance = importance
        self.params = {n: p.clone().detach() for n, p in model.named_parameters()}
        self.fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}

    def update_fisher(self, states, criterion):
        self.model.eval()
        for state in states:
            self.model.zero_grad()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            outputs = self.model(state)
            loss = criterion(outputs, outputs)  # Use outputs as targets for unsupervised learning
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    self.fisher[n] += p.grad ** 2 / len(states)

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self.fisher[n] * (p - self.params[n]) ** 2
            loss += _loss.sum()
        return self.importance * loss
