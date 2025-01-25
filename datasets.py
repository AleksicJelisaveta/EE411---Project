import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
import torchvision
from PIL import Image

class RotatedMNIST(Dataset):
    def __init__(self, data, degree):
        self.data = data
        self.rotation_degree = degree
        self.transform = transforms.Compose([
            transforms.RandomRotation((degree, degree)),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, label = self.data[idx]
        img = self.transform(img)  
        return img, label


class PermutedMNIST(Dataset):
    def __init__(self, data, permutation=None):
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
    

def permute_datasets(train_data, test_data):
    task_datasets_train = []
    task_datasets_test = []
    
    num_tasks = 10
    for i in range(num_tasks):
        random_permutation = np.random.permutation(28 * 28)
        

        permuted_train_dataset = PermutedMNIST(train_data, random_permutation)
        permuted_test_dataset = PermutedMNIST(test_data, random_permutation)
        
        task_datasets_train.append(permuted_train_dataset)
        task_datasets_test.append(permuted_test_dataset)
    
    return task_datasets_train, task_datasets_test


def rotate_datasets(train_data, test_data):
    task_datasets_train = []
    task_datasets_test = []
    for i in range(10):
        degree = 10 * i 
        rotated_data_train = RotatedMNIST(train_data, degree)
        rotated_data_test = RotatedMNIST(test_data, degree)

        task_datasets_train.append(rotated_data_train)
        task_datasets_test.append(rotated_data_test)
    
    return task_datasets_train, task_datasets_test


def create_task_dataloaders(task_datasets, batch_size=64):
    task_loaders = []
    for task_dataset in task_datasets:
        task_loader = DataLoader(task_dataset, batch_size=batch_size, shuffle=True)
        task_loaders.append(task_loader)
    return task_loaders

train_dataset = datasets.MNIST(root='./data', train=True, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, download=True)

permuted_tasks_train, permuted_tasks_test = permute_datasets(train_dataset, test_dataset)

rotated_tasks_train, rotated_tasks_test = rotate_datasets(train_dataset)
