import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from elastic_weight_consolidation import EWC
import datasets as ds

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


def train_model_on_task(model,type, train_dataloader, test_dataloaderA, test_dataloaderB , test_dataloaderC, criterion, optimizer, epochs, ewc=None, lambda_ewc=0.0, early_stopping=None):
    model.train()
    epoch_accuracies_A = []
    epoch_accuracies_B = []
    epoch_accuracies_C = []
    
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
            ewc_loss = ewc.compute_ewc_loss(lambda_ewc) if ewc else 0.0
            loss = task_loss + ewc_loss

            loss.backward()
            optimizer.step()
            total_loss += task_loss.item()
          


        #print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_dataloader):.4f}")
       
       
        # calculate accuracy on test set per epoch


        if type == 'A':
            total = 0
            correct = 0
            for inputs, targets in test_dataloaderA:
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

            epoch_accuracy_A = correct / total    
            epoch_accuracies_A.append(epoch_accuracy_A)
            print(f"Epoch {epoch+1}/{epochs}, Accuracy on test set A: {epoch_accuracy_A:.4f}")

        if type == 'B':
            total = 0
            correct = 0
            for inputs, targets in test_dataloaderB:
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

            epoch_accuracy_B = correct / total    
            epoch_accuracies_B.append(epoch_accuracy_B)
            print(f"Epoch {epoch+1}/{epochs}, Accuracy on test set B: {epoch_accuracy_B:.4f}")  

            total = 0
            correct = 0
            for inputs, targets in test_dataloaderA:
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

            epoch_accuracy_A = correct / total
            epoch_accuracies_A.append(epoch_accuracy_A)
            print(f"Epoch {epoch+1}/{epochs}, Accuracy on test set A: {epoch_accuracy_A:.4f}")

        if type == 'C':
            total = 0
            correct = 0
            for inputs, targets in test_dataloaderC:
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

            epoch_accuracy_C = correct / total    
            epoch_accuracies_C.append(epoch_accuracy_C)
            print(f"Epoch {epoch+1}/{epochs}, Accuracy on test set C: {epoch_accuracy_C:.4f}")  

            total = 0
            correct = 0
            for inputs, targets in test_dataloaderB:
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

            epoch_accuracy_B = correct / total
            epoch_accuracies_B.append(epoch_accuracy_B)
            print(f"Epoch {epoch+1}/{epochs}, Accuracy on test set B: {epoch_accuracy_B:.4f}")  

            total = 0
            correct = 0
            for inputs, targets in test_dataloaderA:
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

            epoch_accuracy_A = correct / total
            epoch_accuracies_A.append(epoch_accuracy_A)
            print(f"Epoch {epoch+1}/{epochs}, Accuracy on test set A: {epoch_accuracy_A:.4f}")
    

        
        

   
    print("\n")  
    return epoch_accuracies_A, epoch_accuracies_B, epoch_accuracies_C   


from sklearn.model_selection import train_test_split

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

# Set hyperparameters
def set_experiment_params(figure_type='2A'):
    if figure_type == '2A':
        learning_rate = 1e-3
        dropout_input = 0.0
        dropout_hidden = 0.0
        early_stopping_enabled = False
        num_hidden_layers = 2
        width_hidden_layers = 400
        epochs = 20
    elif figure_type == '2B':
        learning_rate = np.logspace(-5, -3, 100)
        dropout_input = 0.2
        dropout_hidden = 0.5
        early_stopping_enabled = True
        num_hidden_layers = 2
        width_hidden_layers = range(400,2000)
        epochs = 100
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


def run_experiment_2A(permuted_train_loaders, permuted_test_loaders):
    learning_rate, dropout_input,dropout_hidden, early_stopping_enabled, num_hidden_layers, width_hidden_layers, epochs = set_experiment_params('2A')
    
    print(f"Learning rate: {learning_rate}, Dropout input: {dropout_input}, Dropout hidden: {dropout_hidden}, Early stopping: {early_stopping_enabled}, Num hidden layers: {num_hidden_layers}, Width hidden layers: {width_hidden_layers}, Epochs: {epochs}")

    # Define model
    model = CustomNN(num_hidden_layers=num_hidden_layers, hidden_size=width_hidden_layers, dropout_input=dropout_input, dropout_hidden=dropout_hidden)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Define early stopping
    early_stopping = EarlyStopping(patience=5) if early_stopping_enabled else None

    # Train on first task
    epoch_accuracies_A1, epoch_accuracies_B1, epoch_accuracies_C1 = train_model_on_task(model,'A', permuted_train_loaders[0], permuted_test_loaders[0], [],[], criterion, optimizer, epochs, early_stopping=early_stopping)

    # Train on second task
    epoch_accuracies_A2, epoch_accuracies_B2, epoch_accuracies_C2 = train_model_on_task(model,'B', permuted_train_loaders[1], permuted_test_loaders[0], permuted_test_loaders[1],[], criterion, optimizer, epochs, early_stopping=early_stopping)

    # Train on third task
    epoch_accuracies_A3, epoch_accuracies_B3, epoch_accuracies_C3 = train_model_on_task(model,'C', permuted_train_loaders[2],permuted_test_loaders[0], permuted_test_loaders[1],permuted_test_loaders[2], criterion, optimizer, epochs, early_stopping=early_stopping)

    
    # Define EWC
    ewc = EWC(model, permuted_train_loaders[0])

    # Train on first task with EWC
    epoch_accuracies_A1_ewc, epoch_accuracies_B1_ewc, epoch_accuracies_C1_ewc = train_model_on_task(model,'A', permuted_train_loaders[0], permuted_test_loaders[0], [],[], criterion, optimizer, epochs,ewc = ewc, early_stopping=early_stopping)

    # Train on second task with EWC
    epoch_accuracies_A2_ewc, epoch_accuracies_B2_ewc, epoch_accuracies_C2_ewc = train_model_on_task(model,'B', permuted_train_loaders[1], permuted_test_loaders[0], permuted_test_loaders[1],[], criterion, optimizer, epochs, ewc = ewc, early_stopping=early_stopping)

    # Train on third task with EWC
    epoch_accuracies_A3_ewc, epoch_accuracies_B3_ewc, epoch_accuracies_C3_ewc = train_model_on_task(model,'C', permuted_train_loaders[2],permuted_test_loaders[0], permuted_test_loaders[1],permuted_test_loaders[2], criterion, optimizer, epochs,ewc = ewc, early_stopping=early_stopping)


    # use L2 regularization
    model = CustomNN(num_hidden_layers=num_hidden_layers, hidden_size=width_hidden_layers, dropout_input=dropout_input, dropout_hidden=dropout_hidden)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    early_stopping = EarlyStopping(patience=5) if early_stopping_enabled else None

    # Train on first task with L2 regularization
    epoch_accuracies_A1_L2, epoch_accuracies_B1_L2, epoch_accuracies_C1_L2 = train_model_on_task(model,'A', permuted_train_loaders[0], permuted_test_loaders[0], [],[], criterion, optimizer, epochs, early_stopping=early_stopping)

    # Train on second task with L2 regularization
    epoch_accuracies_A2_L2, epoch_accuracies_B2_L2, epoch_accuracies_C2_L2 = train_model_on_task(model,'B', permuted_train_loaders[1], permuted_test_loaders[0], permuted_test_loaders[1],[], criterion, optimizer, epochs, early_stopping=early_stopping)
    
    # Train on third task with L2 regularization
    epoch_accuracies_A3_L2, epoch_accuracies_B3_L2, epoch_accuracies_C3_L2 = train_model_on_task(model,'C', permuted_train_loaders[2],permuted_test_loaders[0], permuted_test_loaders[1],permuted_test_loaders[2], criterion, optimizer, epochs, early_stopping=early_stopping)


    return epoch_accuracies_A1, epoch_accuracies_B1, epoch_accuracies_C1, epoch_accuracies_A2, epoch_accuracies_B2, epoch_accuracies_C2, epoch_accuracies_A3, epoch_accuracies_B3, epoch_accuracies_C3,  epoch_accuracies_A1_ewc, epoch_accuracies_B1_ewc, epoch_accuracies_C1_ewc, epoch_accuracies_A2_ewc, epoch_accuracies_B2_ewc, epoch_accuracies_C2_ewc, epoch_accuracies_A3_ewc, epoch_accuracies_B3_ewc, epoch_accuracies_C3_ewc, epoch_accuracies_A1_L2, epoch_accuracies_B1_L2, epoch_accuracies_C1_L2, epoch_accuracies_A2_L2, epoch_accuracies_B2_L2, epoch_accuracies_C2_L2, epoch_accuracies_A3_L2, epoch_accuracies_B3_L2, epoch_accuracies_C3_L2



import random
from sklearn.model_selection import train_test_split


class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def run_experiment_2B(train_loaders, criterion, search_trials=50, patience=5):
    """
    Perform random hyperparameter search and train the model on all tasks.

    Args:
        model_class: The class of the model to initialize.
        train_loaders: List of DataLoaders, one for each task.
        criterion: Loss function.
        figure_type: Determines hyperparameter ranges based on experiment type.
        search_trials: Number of random hyperparameter combinations to try.
        patience: Number of epochs for early stopping.
    """
    best_model = None
    best_val_loss = float("inf")
    best_params = None

    learning_rate_range, dropout_input, dropout_hidden, early_stopping_enabled, num_hidden_layers, width_hidden_layer_range, epochs = set_experiment_params('2B')
    # Perform random search over the hyperparameter space
    for trial in range(search_trials):
        # Randomly sample hyperparameters
        learning_rate = random.choice(learning_rate_range)
        hidden_layer_width = random.choice(width_hidden_layer_range)

        print(f"Trial {trial + 1}/{search_trials}: Learning rate={learning_rate:.5f}, "
              f"Hidden layer width={hidden_layer_width}")

        # Initialize the model with sampled hyperparameters
        model = CustomNN(
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_layer_width,
            dropout_input=dropout_input,
            dropout_hidden=dropout_hidden
        )

        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        # Train the model sequentially on tasks
        total_val_loss = 0
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
            early_stopping = EarlyStopping(patience=patience)

            # Training loop for the current task
            for epoch in range(epochs):
                total_loss = 0.0
                model.train()

                # Training phase
                for inputs, targets in train_task_loader:
                    inputs, targets = inputs.to(next(model.parameters()).device), targets.to(next(model.parameters()).device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                print(f"Task {task_id}, Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_task_loader):.4f}")

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
                if early_stopping_enabled and early_stopping(val_loss):
                    print(f"Early stopping triggered on Task {task_id}. Moving to the next task.\n")
                    break

            total_val_loss += val_loss

        # Check if current hyperparameters are the best
        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            best_model = model
            best_params = {
                'learning_rate': learning_rate,
                'hidden_layer_width': hidden_layer_width
            }

    print(f"Best hyperparameters: {best_params}, Best validation loss: {best_val_loss:.4f}")
    return best_model, best_params



def run_experiment_2B_with_ewc(train_loaders, lambda_ewc=100, search_trials=50, patience=5):
 
    criterion = nn.CrossEntropyLoss()
    

    best_sgd_model = None
    best_ewc_model = None
    best_sgd_val_loss = float("inf")
    best_ewc_val_loss = float("inf")
    best_sgd_params = None
    best_ewc_params = None

    # Set experiment parameters for figure type '2B'
    learning_rate_range, dropout_input, dropout_hidden, early_stopping_enabled, num_hidden_layers, width_hidden_layer_range, epochs = set_experiment_params('2B')

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

        ewc = EWC(ewc_model, train_loaders[0])

        sgd_optimizer = optim.SGD(sgd_model.parameters(), lr=learning_rate)
        ewc_optimizer = optim.SGD(ewc_model.parameters(), lr=learning_rate)

        # Train both models sequentially on tasks
        total_sgd_val_loss = 0
        total_ewc_val_loss = 0
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

                sgd_task_loss = 0.0
                ewc_task_loss = 0.0

                # Training phase for SGD and EWC
                for inputs, targets in train_task_loader:
                    inputs, targets = inputs.to(next(sgd_model.parameters()).device), targets.to(next(sgd_model.parameters()).device)

                    # SGD model training
                    sgd_optimizer.zero_grad()
                    sgd_outputs = sgd_model(inputs)
                    loss = criterion(sgd_outputs, targets)
                    loss.backward()
                    sgd_optimizer.step()
                    sgd_task_loss += loss.item()

                    # EWC model training
                    ewc_optimizer.zero_grad()
                    ewc_outputs = ewc_model(inputs)
                    loss = criterion(ewc_outputs, targets)
                    ewc_loss = loss + ewc.compute_ewc_loss(lambda_ewc)
                    ewc_loss.backward()
                    ewc_optimizer.step()
                    ewc_task_loss += loss.item()

                print(f"Task {task_id}, Epoch {epoch + 1}/{epochs}, SGD Loss: {sgd_task_loss / len(train_task_loader):.4f}, "
                      f"EWC Loss: {ewc_task_loss / len(train_task_loader):.4f}")

                # Validation phase for both models
                sgd_val_loss = 0.0
                ewc_val_loss = 0.0
                sgd_model.eval()
                ewc_model.eval()
                with torch.no_grad():
                    for inputs, targets in val_task_loader:
                        inputs, targets = inputs.to(next(sgd_model.parameters()).device), targets.to(next(sgd_model.parameters()).device)

                        # SGD model validation
                        sgd_outputs = sgd_model(inputs)
                        sgd_val_loss += criterion(sgd_outputs, targets).item()

                        # EWC model validation
                        ewc_outputs = ewc_model(inputs)
                        ewc_val_loss += criterion(ewc_outputs, targets).item()
                sgd_val_loss /= len(val_task_loader)
                ewc_val_loss /= len(val_task_loader)

                print(f"Validation Loss on Task {task_id}, Epoch {epoch + 1}, SGD: {sgd_val_loss:.4f}, EWC: {ewc_val_loss:.4f}")

                # Check for early stopping
                if early_stopping_enabled and early_stopping_sgd(sgd_val_loss):
                    print(f"SGD Early stopping triggered on Task {task_id}. Moving to the next task.\n")
                    break

                if early_stopping_enabled and early_stopping_ewc(ewc_val_loss):
                    print(f"EWC Early stopping triggered on Task {task_id}. Moving to the next task.\n")
                    break

            total_sgd_val_loss += sgd_val_loss
            total_ewc_val_loss += ewc_val_loss

        # Update best models and parameters if current trial is better
        if total_sgd_val_loss < best_sgd_val_loss:
            best_sgd_val_loss = total_sgd_val_loss
            best_sgd_model = sgd_model
            best_sgd_params = {
                'learning_rate': learning_rate,
                'hidden_layer_width': hidden_layer_width
            }

        if total_ewc_val_loss < best_ewc_val_loss:
            best_ewc_val_loss = total_ewc_val_loss
            best_ewc_model = ewc_model
            best_ewc_params = {
                'learning_rate': learning_rate,
                'hidden_layer_width': hidden_layer_width
            }

    print(f"Best SGD parameters: {best_sgd_params}, Best SGD validation loss: {best_sgd_val_loss:.4f}")
    print(f"Best EWC parameters: {best_ewc_params}, Best EWC validation loss: {best_ewc_val_loss:.4f}")

    return best_sgd_model, best_sgd_params, best_ewc_model, best_ewc_params


# ovde treba da se edituje da se doda drugi data loader koji ima vecu i manju permutaciju
def run_experiment_2C(permuted_train_loaders, permuted_test_loaders):
    learning_rate, dropout_input,dropout_hidden, early_stopping_enabled, num_hidden_layers, width_hidden_layers, epochs = set_experiment_params('2C')
    
    print(f"Learning rate: {learning_rate}, Dropout input: {dropout_input}, Dropout hidden: {dropout_hidden}, Early stopping: {early_stopping_enabled}, Num hidden layers: {num_hidden_layers}, Width hidden layers: {width_hidden_layers}, Epochs: {epochs}")

    # Define model
    model = CustomNN(num_hidden_layers=num_hidden_layers, hidden_size=width_hidden_layers, dropout_input=dropout_input, dropout_hidden=dropout_hidden)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=5) if early_stopping_enabled else None


    # Define EWC
    ewc = EWC(model, permuted_train_loaders[0])

    # Train on first task with EWC
    train_model_on_task(model, permuted_train_loaders[0], criterion, optimizer, epochs, ewc=ewc, lambda_ewc=1e3, early_stopping=early_stopping)

    # Evaluate on first task with EWC
    accuracy11_ewc = evaluate_model_on_task(model, permuted_test_loaders[0])
    print(f"Accuracy on task 1 with EWC: {accuracy11_ewc:.4f}\n")

    # Train on second task with EWC
    train_model_on_task(model, permuted_train_loaders[1], criterion, optimizer, epochs, ewc=ewc, lambda_ewc=1e3, early_stopping=early_stopping)

    # Evaluate on second task with EWC
    accuracy22_ewc = evaluate_model_on_task(model, permuted_test_loaders[1])
    print(f"Accuracy on task 2 with EWC: {accuracy22_ewc:.4f}")
    accuracy21_ewc = evaluate_model_on_task(model, permuted_test_loaders[0])
    print(f"Accuracy on task 1 with EWC: {accuracy21_ewc:.4f}\n")

    # Train on third task with EWC
    train_model_on_task(model, permuted_train_loaders[2], criterion, optimizer, epochs, ewc=ewc, lambda_ewc=1e3, early_stopping=early_stopping)

    # Evaluate on third task with EWC
    accuracy33_ewc = evaluate_model_on_task(model, permuted_test_loaders[2])
    print(f"Accuracy on task 3 with EWC: {accuracy33_ewc:.4f}")
    accuracy32_ewc = evaluate_model_on_task(model, permuted_test_loaders[1])
    print(f"Accuracy on task 2 with EWC: {accuracy32_ewc:.4f}")
    accuracy31_ewc = evaluate_model_on_task(model, permuted_test_loaders[0])
    print(f"Accuracy on task 1 with EWC: {accuracy31_ewc:.4f}\n")


  

    return  accuracy11_ewc, accuracy22_ewc, accuracy21_ewc, accuracy33_ewc, accuracy32_ewc, accuracy31_ewc


