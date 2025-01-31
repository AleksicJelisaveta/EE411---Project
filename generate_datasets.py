import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

class RotatedMNIST(Dataset):
    """
    Custom Dataset for RotatedMNIST

    Args:
        data: The original MNIST dataset (as NumPy arrays).
        data: The original MNIST dataset (as NumPy arrays).
        degree: The angle (in degrees) to rotate the images.
    """
    def __init__(self, data, degree):
        self.data = data
        self.rotation_degree = degree  

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]

        img = np.array(img, dtype=np.float32)

        # normalization
        img = (img - np.min(img)) / (np.max(img) - np.min(img))

        rotated_image = rotate(img, self.rotation_degree, reshape = False)

        rotated_image_tensor = torch.tensor(rotated_image, dtype=torch.float32).unsqueeze(0)  

        return rotated_image_tensor, label


class PermutedMNIST(Dataset):
    """
    Custom Dataset for PermutedMNIST with optional normalization.

    Args:
        data: The original MNIST dataset.
        permutation: A predefined pixel permutation (optional).
                     If None, a random permutation is generated.
    """
    def __init__(self, data, permutation=None, normalize=True):
        self.data = data
        self.permutation = permutation if permutation is not None else np.random.permutation(28*28)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]

        img_flattened = np.array(img).flatten()  
        permuted_image = img_flattened[self.permutation] 
        permuted_image = permuted_image.reshape(28, 28)  

        permuted_image_tensor = torch.tensor(permuted_image, dtype=torch.float32).unsqueeze(0)  

        return permuted_image_tensor, label



def permute_datasets(train_data, test_data, num_tasks=3, base_seed=42):
    """
    Generates multiple permuted datasets for train and test sets.
    
    Args:
        train_data: Training dataset.
        test_data: Testing dataset.
        num_tasks: Number of permuted tasks.
        base_seed: Base seed to ensure different permutations across tasks.
    
    Returns:
        task_datasets_permute_train: List of permuted training datasets.
        task_datasets_permute_test: List of permuted testing datasets.
    """
    task_datasets_permute_train = []
    task_datasets_permute_test = []

    for i in range(num_tasks):
        # Set a different seed for each task to ensure unique permutation
        task_seed = base_seed + i
        rng = np.random.default_rng(task_seed)
        
        # Generate a fixed permutation for both train and test sets
        random_permutation = rng.permutation(28 * 28)
        
        # Apply the same permutation for both train and test sets
        permuted_train_dataset = PermutedMNIST(train_data, random_permutation)
        permuted_test_dataset = PermutedMNIST(test_data, random_permutation)
        
        task_datasets_permute_train.append(permuted_train_dataset)
        task_datasets_permute_test.append(permuted_test_dataset)
    
    return task_datasets_permute_train, task_datasets_permute_test


def rotate_datasets(train_data, test_data, num_tasks=10, base_seed=42):
    """ 
    Generates multiple rotated datasets for train and test sets.
    
    Args:
        train_data: Training dataset.
        test_data: Testing dataset.
        num_tasks: Number of rotated tasks.
        base_seed: Base seed to ensure different rotation degrees across tasks.
    
    Returns:
        task_datasets_rotate_train: List of rotated training datasets.
        task_datasets_rotate_test: List of rotated testing datasets.
    """   
    task_datasets_rotate_train = []
    task_datasets_rotate_test = []

    for i in range(num_tasks):
        # Set the degree of rotation based on the task index
        degree = 10 * i  # Example: task i rotates by 10*i degrees
        
        # Apply the same rotation for both train and test sets
        rotated_data_train = RotatedMNIST(train_data, degree)
        rotated_data_test = RotatedMNIST(test_data, degree)

        task_datasets_rotate_train.append(rotated_data_train)
        task_datasets_rotate_test.append(rotated_data_test)
    
    return task_datasets_rotate_train, task_datasets_rotate_test


def create_task_dataloaders(task_datasets, batch_size=64):
    """
    Converts datasets into DataLoaders for batching and shuffling.
    
    Args:
        task_datasets: List of datasets for different tasks.
        batch_size: Number of samples per batch.
    
    Returns:
        task_loaders: List of DataLoaders for the tasks.
    """
    task_loaders = []
    for task_dataset in task_datasets:
        task_loader = DataLoader(task_dataset, batch_size=batch_size, shuffle=True)
        task_loaders.append(task_loader)
    return task_loaders


def load_datasets(num_tasks_permute=3, num_tasks_rotate=10, base_seed=42):
    """
    Loads the MNIST dataset and generates permuted and rotated tasks.
    Creates DataLoaders for these tasks.
    
    Args:
        num_tasks_permute: Number of permuted tasks.
        num_tasks_rotate: Number of rotated tasks.
        base_seed: Base seed for reproducibility.
    
    Returns:
        permuted_train_loaders: List of DataLoaders for permuted training tasks.
        permuted_test_loaders: List of DataLoaders for permuted testing tasks.
        rotated_train_loaders: List of DataLoaders for rotated training tasks.
        rotated_test_loaders: List of DataLoaders for rotated testing tasks.
    """
    train_dataset = datasets.MNIST(root='./data', train=True, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True)

    # Generate permuted datasets for both train and test
    permuted_tasks_train, permuted_tasks_test = permute_datasets(train_dataset, test_dataset, num_tasks=num_tasks_permute, base_seed=base_seed)

    # Generate rotated datasets for both train and test
    rotated_tasks_train, rotated_tasks_test = rotate_datasets(train_dataset, test_dataset, num_tasks=num_tasks_rotate, base_seed=base_seed)

    # Create data loaders for permuted datasets
    permuted_train_loaders = create_task_dataloaders(permuted_tasks_train)
    permuted_test_loaders = create_task_dataloaders(permuted_tasks_test)

    # Create data loaders for rotated datasets
    rotated_train_loaders = create_task_dataloaders(rotated_tasks_train)
    rotated_test_loaders = create_task_dataloaders(rotated_tasks_test)

    return permuted_train_loaders, permuted_test_loaders, rotated_train_loaders, rotated_test_loaders

