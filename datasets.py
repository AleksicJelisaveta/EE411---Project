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
        if permutation is not None:
            self.permutation = permutation
        else:
            np.random.permutation(28*28)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, label = self.data[idx]
        
        img_flattened = np.array(img).flatten()  
        permuted_image = img_flattened[self.permutation] 
        permuted_image = permuted_image.reshape(28, 28)  
        
        permuted_image_tensor = torch.tensor(permuted_image, dtype=torch.float32).unsqueeze(0)  
        
        return permuted_image_tensor, label
    

    #Permute each task 
def permute_dataset(data):

    task_datasets_permuted = []
    num_tasks = 10
    for i in range(num_tasks): 
        random_permutation = np.random.permutation(28 * 28) 
        permuted_dataset = PermutedMNIST(data, random_permutation)
        task_datasets_permuted.append(permuted_dataset)
    
    return task_datasets_permuted


#Rotate each task
def rotate_dataset(data):

    task_datasets_rotated = []
    for i in range(10):
        degree = 10 * i 
        rotated_data = RotatedMNIST(data, degree)
        task_datasets_rotated.append(rotated_data)
    
    return task_datasets_rotated



def create_task_dataloaders(task_datasets, batch_size=64):
    task_loaders = []
    for task_dataset in task_datasets:
        task_loader = DataLoader(task_dataset, batch_size=batch_size, shuffle=True)
        task_loaders.append(task_loader)
    return task_loaders

#Load MNIST dataset for train and test
train_dataset = datasets.MNIST(root='./data', train=True, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, download=True)

#Permute train and test sets
permuted_tasks_train = permute_dataset(train_dataset)
permuted_tasks_test = permute_dataset(test_dataset)

#Rotate train and test sets
rotated_tasks_train = rotate_dataset(train_dataset)
rotated_tasks_test = rotate_dataset(test_dataset)