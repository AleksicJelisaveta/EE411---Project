import torch

class ElasticWeightConsolidationAtari:
    # Elastic Weight Consolidation (EWC) for Atari environments for ddqn agent
    def __init__(self, model):
        """
        Initialize the EWC object for Atari environments.

        Args:
            model (torch.nn.Module): The model to regularize.
        """
        self.model = model
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}  # Mean parameter values from the previous task
        self._precision_matrices = {n: torch.zeros_like(p) for n, p in self.params.items()}

    def compute_fisher(self, data_loader):
        """
        Compute the Fisher Information Matrix for the current task.

        Args:
            data_loader (torch.utils.data.DataLoader): DataLoader for the dataset of the current task.
        """
        self.model.eval()
        
        # Reset precision matrix before accumulation
        for n in self._precision_matrices:
            self._precision_matrices[n].zero_()

        total_samples = 0
        
        for inputs, _ in data_loader:
            self.model.zero_grad()
            
            outputs = self.model(inputs)
            loss = outputs  # There is no loss as it is reinforcement learning
            loss.backward()
            
            batch_size = inputs.shape[0]
            total_samples += batch_size

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    self._precision_matrices[n] += (p.grad ** 2) * batch_size  # Weight by batch size
        
        # Normalize over total samples
        for n in self._precision_matrices:
            self._precision_matrices[n] /= total_samples

    def update_params(self):
        """
        Store the mean parameter values after training on a task.
        """
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self._means[n] = p.clone().detach()

    def compute_ewc_loss(self, model, lambda_ewc=10.0):
        """
        Compute EWC loss for the current model parameters.

        Args:
            model (torch.nn.Module): The current model.
            lambda_ewc (float): Regularization strength for EWC.

        Returns:
            torch.Tensor: The EWC regularization loss.
        """
        if not self._means:
            return torch.tensor(0.0)
        
        loss = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
                loss += _loss.sum()

        return lambda_ewc * loss

