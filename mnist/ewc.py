import torch
import torch.nn.functional as F

class EWC:
    def __init__(self, model):
        self.model = model
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}  # Mean parameter values from the previous task
        self._precision_matrices = {n: torch.zeros_like(p) for n, p in self.params.items()}

    def compute_fisher(self, data_loader):
        """Compute the Fisher Information Matrix for the current task."""
        self.model.eval()
        
        # Reset precision matrix before accumulation
        # for n in self._precision_matrices:
        #     self._precision_matrices[n].zero_()
        
        for inputs, labels in data_loader:
            self.model.zero_grad()
            
            outputs = self.model(inputs)
            loss = F.nll_loss(F.log_softmax(outputs, dim=1), labels)
            loss.backward()
            
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    self._precision_matrices[n] += (p.grad ** 2) / len(data_loader)

    def update_params(self):
        """Store the mean parameter values after training on a task."""
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self._means[n] = p.clone().detach()

    def compute_ewc_loss(self, model, lambda_ewc=10.0):
        """Compute EWC loss for the current model parameters."""
        if not self._means:
            return torch.tensor(0.0)  # No previous tasks, no regularization needed
        
        loss = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
                loss += _loss.sum()
        
        return lambda_ewc * loss
