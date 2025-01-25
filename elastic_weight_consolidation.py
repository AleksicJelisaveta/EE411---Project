import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np


class EWC:
    def __init__(self, model, dataloader, device='cpu'):
        """
        Elastic Weight Consolidation (EWC) implementation.

        Args:
            model (torch.nn.Module): The model to regularize.
            dataloader (torch.utils.data.DataLoader): DataLoader for the dataset of the previous task.
            device (str): Device to perform computations on ('cpu' or 'cuda').
        """
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = None
        self.prev_params = None

    def compute_fisher(self):
        """
        Compute the Fisher Information Matrix for the current task.
        """
        # Initialize Fisher Information matrix
        fisher = {n: torch.zeros_like(p, device=self.device) for n, p in self.params.items()}

        # Set the model to evaluation mode
        self.model.eval()

        # Loop through the dataloader and accumulate Fisher Information
        for inputs, targets in self.dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            # reshape inputs to (batch_size, input_dim * input_dim) from (batch_size, 1, input_dim, input_dim)
            inputs = inputs.view(inputs.size(0), -1)

            # Zero gradients
            self.model.zero_grad()

            # Forward pass and compute log likelihood
            outputs = self.model(inputs)
            log_likelihood = torch.nn.functional.log_softmax(outputs, dim=1)
            loss = log_likelihood[range(len(targets)), targets].mean()

            # Backward pass to compute gradients
            loss.backward()

            # Accumulate Fisher Information from gradients
            for n, p in self.params.items():
                fisher[n] += p.grad ** 2 / len(self.dataloader)

        self.fisher = fisher

    def store_prev_params(self):
        """
        Store the current parameters of the model.
        """
        self.prev_params = {n: p.clone().detach() for n, p in self.params.items()}

    def compute_ewc_loss(self, lambda_ewc):
        """
        Compute the EWC regularization loss.

        Args:
            lambda_ewc (float): Regularization strength for EWC.

        Returns:
            torch.Tensor: The EWC regularization loss.
        """
        if self.fisher is None or self.prev_params is None:
            return 0.0

        ewc_loss = 0.0
        for n, p in self.params.items():
            fisher_term = self.fisher[n]
            prev_param = self.prev_params[n]
            ewc_loss += (fisher_term * (p - prev_param) ** 2).sum()

        return lambda_ewc * ewc_loss