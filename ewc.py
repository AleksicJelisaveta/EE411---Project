import torch
import torch.nn.functional as F

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EWC:
    """
    This module implements the Elastic Weight Consolidation (EWC) algorithm for continual learning.
    EWC helps to mitigate catastrophic forgetting when training neural networks on sequential tasks.
    The implementation is based on the original paper: "Overcoming catastrophic forgetting in neural networks" by Kirkpatrick et al.
    Classes:
        EWC: A class that encapsulates the EWC algorithm, including methods to compute the Fisher Information Matrix,
             update parameter means, and compute the EWC loss.
    Methods:
        __init__(self, model):
            Initializes the EWC object with the given model, and sets up the necessary data structures.
        compute_fisher(self, data_loader):
            Computes the Fisher Information Matrix for the current task using the provided data loader.
        update_params(self):
            Stores the mean parameter values after training on a task.
        compute_ewc_loss(self, model, lambda_ewc=10.0):
            Computes the EWC loss for the current model parameters, which acts as a regularization term to prevent
            catastrophic forgetting.
    """
    def __init__(self, model):
        self.model = model.to(device)
        self.params = {n: p.to(device) for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}  # Mean parameter values from the previous task
        self._precision_matrices = {n: torch.zeros_like(p).to(device) for n, p in self.params.items()}

    def compute_fisher(self, data_loader):
        """Compute the Fisher Information Matrix for the current task."""
        self.model.eval()
        
        # Reset precision matrix before accumulation
        # for n in self._precision_matrices:
        #     self._precision_matrices[n].zero_()
        
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
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
            return torch.tensor(0.0, device = device)  # No previous tasks, no regularization needed
        
        loss = torch.tensor(0.0, device = device)
        for n, p in model.named_parameters():
            if p.requires_grad:
                _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
                loss += _loss.sum()
        
        return lambda_ewc * loss
