import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EWC:
    def __init__(self, model: torch.nn.Module):
        self.model = model.to(device)
        self.params = {n: p.to(device) for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}  # Mean parameter values from the previous task
        self._precision_matrices = {n: torch.zeros_like(p).to(device) for n, p in self.params.items()}

    def compute_fisher(self, data_loader):
        """Compute the Fisher Information Matrix for the current task."""
        self.model.eval()
        
        # Reset precision matrix before accumulation
        for n in self._precision_matrices:
            self._precision_matrices[n].zero_()

        total_samples = 0
        
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            self.model.zero_grad()
            
            outputs = self.model(inputs)
            loss = F.nll_loss(F.log_softmax(outputs, dim=1), labels)  # Ensure log-probs for nll_loss
            loss.backward()
            
            batch_size = inputs.shape[0]
            total_samples += batch_size

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    self._precision_matrices[n] += (p.grad ** 2).detach() * batch_size  # Weight by batch size
        
        # Normalize over total samples
        for n in self._precision_matrices:
            self._precision_matrices[n] /= total_samples

    def update_params(self):
        """Store the mean parameter values after training on a task."""
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self._means[n] = p.clone().detach()

    def compute_ewc_loss(self, model: torch.nn.Module, lambda_ewc: float=10.0)-> torch.Tensor:
        """Compute EWC loss for the current model parameters."""
        if not self._means:
            return torch.tensor(0.0, device = device)  # No previous tasks, no regularization needed
        
        loss = torch.tensor(0.0, device = device)
        for n, p in model.named_parameters():
            if p.requires_grad and n in self._means and n in self._precision_matrices:
                _loss = 0.5*self._precision_matrices[n] * (p - self._means[n]) ** 2
                loss += _loss.sum()
        
        return lambda_ewc * loss
