import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from moja_ewc import EWC
import generate_datasets as gen_ds
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt

class CustomNN(nn.Module):
    def __init__(self, num_hidden_layers=2, hidden_size=400, input_size=28*28, output_size=10, dropout_input=0.2, dropout_hidden=0.5):
        super(CustomNN, self).__init__()
        layers = []
        in_features = input_size

        # Add input layer with dropout
        if dropout_input > 0:
            layers.append(nn.Dropout(dropout_input))
        layers.append(nn.Linear(in_features, hidden_size))
        layers.append(nn.ReLU())

        # Add hidden layers with dropout
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            if dropout_hidden > 0:
                layers.append(nn.Dropout(dropout_hidden))

        # Output layer
        layers.append(nn.Linear(hidden_size, output_size))

        # Combine all layers
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x.view(x.size(0), -1))  # Flatten input


def train_model_on_task(model, train_type_id, train_dataloader, test_dataloader, criterion, optimizer, epochs, ewc=None, lambda_ewc=0.0, early_stopping=None):
    model.train()
    
    accuracies = {}

    for id in range(train_type_id):
        accuracies[id] = []

    for epoch in range(epochs):
        total_loss = 0
        
        # for each epoch print accuracy
        accuracy = 0
        total = 0
        correct = 0

        for inputs, targets in train_dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            task_loss = criterion(outputs, targets)

            # Add regularization loss if applicable
            ewc_loss = ewc.compute_ewc_loss(model,lambda_ewc) if ewc else 0.0
            loss = task_loss + ewc_loss

            loss.backward()
            optimizer.step()
            total_loss += task_loss.item()
          
       
       
        # calculate accuracy on test set per epoch

        model.eval()
        with torch.no_grad():

            for id in range(train_type_id):
                total = 0
                correct = 0

                for inputs, targets in test_dataloader[id]:
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()

                epoch_accuracy = correct / total
                accuracies[id].append(epoch_accuracy)
                print(f"Epoch {epoch+1}/{epochs}, Accuracy on test set {id}: {epoch_accuracy:.4f}")
                            
   
    print("\n")  
    return accuracies


class EarlyStopping:
    def __init__(self, patience=5):
        """
        Early stopping class to monitor validation loss and stop training when it doesn't improve.
        Args:
            patience (int): Number of epochs to wait for validation loss improvement before stopping.
        """
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0  # Reset counter if validation loss improves
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
        
    # Set hyperparameters
def set_experiment_params(figure_type='2A'):
    if figure_type == '2A':
        learning_rate = 1e-3
        dropout_input = 0.0
        dropout_hidden = 0.0
        early_stopping_enabled = False
        num_hidden_layers = 2
        width_hidden_layers = 400
        epochs = 3
    elif figure_type == '2B':
        learning_rate = np.logspace(-5, -3, 100)
        dropout_input = 0.2
        dropout_hidden = 0.5
        early_stopping_enabled = True
        num_hidden_layers = 2
        width_hidden_layers = range(400,2000)
        epochs = 10
    elif figure_type == '2C':
        learning_rate = 1e-3
        dropout_input = 0.0
        dropout_hidden = 0.0
        early_stopping_enabled = False
        num_hidden_layers = 6
        width_hidden_layers = 100
        epochs = 100
    else:
        raise ValueError(f"Unknown figure type: {figure_type}")
    
    return learning_rate, dropout_input,dropout_hidden, early_stopping_enabled, num_hidden_layers, width_hidden_layers, epochs


def train_model_on_tasks(model, train_loaders, criterion, optimizer, epochs_per_task, ewc=None, lambda_ewc=0.0, patience=5):
    """
    Train the model sequentially on a list of tasks, transitioning to the next task when early stopping is triggered.

    Args:
        model: Neural network model.
        train_loaders: List of DataLoaders, where each DataLoader corresponds to a task.
        criterion: Loss function.
        optimizer: Optimizer.
        epochs_per_task: Number of epochs allocated for each task.
        ewc: Elastic Weight Consolidation object (optional).
        lambda_ewc: Regularization strength for EWC.
        patience: Number of epochs for early stopping.
    """
    for task_id, task_loader in enumerate(train_loaders, 1):
        print(f"Training on Task {task_id}")

        # Split task_loader dataset into training and validation sets
        task_indices = list(range(len(task_loader.dataset)))
        train_indices, val_indices = train_test_split(task_indices, test_size=0.2, shuffle=True)
        train_subset = Subset(task_loader.dataset, train_indices)
        val_subset = Subset(task_loader.dataset, val_indices)
        train_task_loader = DataLoader(train_subset, batch_size=task_loader.batch_size, shuffle=True)
        val_task_loader = DataLoader(val_subset, batch_size=task_loader.batch_size, shuffle=False)

        # Initialize early stopping
        early_stopping = EarlyStopping(patience=patience)

        # Training loop for the current task
        for epoch in range(epochs_per_task):
            total_loss = 0.0
            model.train()

            # Training phase
            for inputs, targets in train_task_loader:
                inputs, targets = inputs.to(next(model.parameters()).device), targets.to(next(model.parameters()).device)
                optimizer.zero_grad()
                outputs = model(inputs)
                task_loss = criterion(outputs, targets)

                # Add EWC loss if applicable
                ewc_loss = ewc.compute_ewc_loss(lambda_ewc) if ewc else 0.0
                loss = task_loss + ewc_loss

                loss.backward()
                optimizer.step()
                total_loss += task_loss.item()

            print(f"Task {task_id}, Epoch {epoch + 1}/{epochs_per_task}, Loss: {total_loss / len(train_task_loader):.4f}")

            # Validation phase
            val_loss = 0.0
            model.eval()
            with torch.no_grad():
                for inputs, targets in val_task_loader:
                    inputs, targets = inputs.to(next(model.parameters()).device), targets.to(next(model.parameters()).device)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, targets).item()
            val_loss /= len(val_task_loader)
            print(f"Validation Loss on Task {task_id}, Epoch {epoch + 1}: {val_loss:.4f}")

            # Check for early stopping
            if early_stopping(val_loss):
                print(f"Early stopping triggered on Task {task_id}. Moving to the next task.\n")
                break


# Evaluate model after training on a task
def evaluate_model_on_task(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    return correct / total


def run_experiment_2A(permuted_train_loaders, permuted_test_loaders):
    learning_rate, dropout_input,dropout_hidden, early_stopping_enabled, num_hidden_layers, width_hidden_layers, epochs = set_experiment_params('2A')
    
    print(f"Learning rate: {learning_rate}, Dropout input: {dropout_input}, Dropout hidden: {dropout_hidden}, Early stopping: {early_stopping_enabled}, Num hidden layers: {num_hidden_layers}, Width hidden layers: {width_hidden_layers}, Epochs: {epochs}")

    # Define model
    model = CustomNN(num_hidden_layers=num_hidden_layers, hidden_size=width_hidden_layers, dropout_input=dropout_input, dropout_hidden=dropout_hidden)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Define early stopping
    early_stopping = EarlyStopping(patience=5) if early_stopping_enabled else None

    epoch_accuracies_SGD = {}
    epoch_accuracies_EWC = {}
    epoch_accuracies_L2 = {}

    for task_num in range(3):
        epoch_accuracies_SGD[task_num] = train_model_on_task(model, task_num+1, permuted_train_loaders[task_num], permuted_test_loaders[0:task_num+1], criterion, optimizer, epochs, early_stopping=early_stopping)
    
    print(epoch_accuracies_SGD)

    # Define EWC
    model_ewc = CustomNN(num_hidden_layers=num_hidden_layers, hidden_size=width_hidden_layers, dropout_input=dropout_input, dropout_hidden=dropout_hidden)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_ewc.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=5) if early_stopping_enabled else None

    # Train on first task with EWC
    ewc = EWC(model_ewc)

    for task_num in range(3):
        lambda_ewc = 500 if task_num != 0 else None
        epoch_accuracies_EWC[task_num] = train_model_on_task(model_ewc, task_num+1, permuted_train_loaders[task_num], permuted_test_loaders[0:task_num+1], criterion, optimizer, epochs, ewc=ewc, lambda_ewc=lambda_ewc, early_stopping=early_stopping)
        
        ewc.compute_fisher(permuted_train_loaders[task_num])
        ewc.update_params()

    
    # Use L2 regularization
    model_l2 = CustomNN(num_hidden_layers=num_hidden_layers, hidden_size=width_hidden_layers, dropout_input=dropout_input, dropout_hidden=dropout_hidden)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_l2.parameters(), lr=learning_rate, weight_decay=1e-3)
    early_stopping = EarlyStopping(patience=5) if early_stopping_enabled else None

    for task_num in range(3):
        epoch_accuracies_L2[task_num] = train_model_on_task(model_l2, task_num+1, permuted_train_loaders[task_num], permuted_test_loaders[0:task_num+1], criterion, optimizer, epochs, early_stopping=early_stopping)
    
    return epoch_accuracies_SGD, epoch_accuracies_EWC, epoch_accuracies_L2


def run_experiment_2B(train_loaders, test_loaders,lambda_ewc=100, search_trials=5, patience=5):
    
        criterion = nn.CrossEntropyLoss()

        # Experiment Parameters
        learning_rate_range, dropout_input, dropout_hidden, early_stopping_enabled, num_hidden_layers, width_hidden_layer_range, epochs = set_experiment_params('2B')

        # Variables to track results for plotting
        sgd_accuracies = []
        ewc_accuracies = []
        num_tasks = len(train_loaders)

        for trial in range(search_trials):
            # Randomly sample hyperparameters
            learning_rate = random.choice(learning_rate_range)
            hidden_layer_width = random.choice(width_hidden_layer_range)

            print(f"Trial {trial + 1}/{search_trials}: Learning rate={learning_rate:.5f}, "
                  f"Hidden layer width={hidden_layer_width}")

            # Initialize models for SGD and EWC
            sgd_model = CustomNN(
                num_hidden_layers=num_hidden_layers,
                hidden_size=hidden_layer_width,
                dropout_input=dropout_input,
                dropout_hidden=dropout_hidden
            )

            ewc_model = CustomNN(
                num_hidden_layers=num_hidden_layers,
                hidden_size=hidden_layer_width,
                dropout_input=dropout_input,
                dropout_hidden=dropout_hidden
            )

            ewc = EWC(ewc_model)

            sgd_optimizer = optim.SGD(sgd_model.parameters(), lr=learning_rate)
            ewc_optimizer = optim.SGD(ewc_model.parameters(), lr=learning_rate)

            # Task training loop
            sgd_fraction_correct = []
            ewc_fraction_correct = []

            for task_id, task_loader in enumerate(train_loaders, 1):
                print(f"Training on Task {task_id}")

                # Split task_loader dataset into train and validation sets
                task_indices = list(range(len(task_loader.dataset)))
                train_indices, val_indices = train_test_split(task_indices, test_size=0.2, shuffle=True)
                train_subset = Subset(task_loader.dataset, train_indices)
                val_subset = Subset(task_loader.dataset, val_indices)
                train_task_loader = DataLoader(train_subset, batch_size=task_loader.batch_size, shuffle=True)
                val_task_loader = DataLoader(val_subset, batch_size=task_loader.batch_size, shuffle=False)

                # Initialize early stopping
                early_stopping_sgd = EarlyStopping(patience=patience)
                early_stopping_ewc = EarlyStopping(patience=patience)

                for epoch in range(epochs):
                    sgd_model.train()
                    ewc_model.train()

                    # Training phase
                    for inputs, targets in train_task_loader:
                        inputs, targets = inputs.to(next(sgd_model.parameters()).device), targets.to(next(sgd_model.parameters()).device)

                        # SGD model training
                        sgd_optimizer.zero_grad()
                        sgd_outputs = sgd_model(inputs)
                        sgd_loss = criterion(sgd_outputs, targets)
                        sgd_loss.backward()
                        sgd_optimizer.step()

                        # EWC model training
                        ewc_optimizer.zero_grad()
                        ewc_outputs = ewc_model(inputs)
                        ewc_loss = criterion(ewc_outputs, targets) + ewc.compute_ewc_loss(ewc_model,lambda_ewc)
                        ewc_loss.backward()
                        ewc_optimizer.step()

                    print(f"Task {task_id}, Epoch {epoch + 1}/{epochs}, SGD Loss: {sgd_loss / len(train_task_loader):.4f}")
                    print(f"Task {task_id}, Epoch {epoch + 1}/{epochs}, EWC Loss: {ewc_loss / len(train_task_loader):.4f}")


                    # Validation phase
                    sgd_correct, ewc_correct = 0, 0
                    sgd_model.eval()
                    ewc_model.eval()
                    with torch.no_grad():
                        for inputs, targets in val_task_loader:
                            inputs, targets = inputs.to(next(sgd_model.parameters()).device), targets.to(next(sgd_model.parameters()).device)

                            # SGD model validation
                            sgd_outputs = sgd_model(inputs)
                            sgd_correct += (sgd_outputs.argmax(dim=1) == targets).sum().item()

                            # EWC model validation
                            ewc_outputs = ewc_model(inputs)
                            ewc_correct += (ewc_outputs.argmax(dim=1) == targets).sum().item()

                    sgd_val_accuracy = sgd_correct / len(val_subset)
                    ewc_val_accuracy = ewc_correct / len(val_subset)

                    # Check early stopping
                    if early_stopping_enabled and early_stopping_sgd(1 - sgd_val_accuracy):
                        print(f"SGD Early stopping triggered on Task {task_id}. Moving to the next task.\n")
                        break

                    if early_stopping_enabled and early_stopping_ewc(1 - ewc_val_accuracy):
                        print(f"EWC Early stopping triggered on Task {task_id}. Moving to the next task.\n")
                        break

                # Evaluate performance on all seen tasks
                sgd_task_accuracy = []
                ewc_task_accuracy = []
                for prev_task_id, prev_task_loader in enumerate(test_loaders[:task_id], 1):
                    prev_task_correct_sgd, prev_task_correct_ewc = 0, 0
                    prev_task_total = 0
                    with torch.no_grad():
                        for inputs, targets in prev_task_loader:
                            inputs, targets = inputs.to(next(sgd_model.parameters()).device), targets.to(next(sgd_model.parameters()).device)

                            sgd_outputs = sgd_model(inputs)
                            prev_task_correct_sgd += (sgd_outputs.argmax(dim=1) == targets).sum().item()

                            ewc_outputs = ewc_model(inputs)
                            prev_task_correct_ewc += (ewc_outputs.argmax(dim=1) == targets).sum().item()

                            prev_task_total += targets.size(0)

                    sgd_task_accuracy.append(prev_task_correct_sgd / prev_task_total)
                    ewc_task_accuracy.append(prev_task_correct_ewc / prev_task_total)

                sgd_fraction_correct.append(np.mean(sgd_task_accuracy))
                ewc_fraction_correct.append(np.mean(ewc_task_accuracy))

                ewc.compute_fisher(task_loader)
                ewc.update_params()

            sgd_accuracies.append(sgd_fraction_correct)
            ewc_accuracies.append(ewc_fraction_correct)


        avg_sgd_accuracy = np.mean(sgd_accuracies, axis=0)
        avg_ewc_accuracy = np.mean(ewc_accuracies, axis=0)


        return avg_sgd_accuracy, avg_ewc_accuracy