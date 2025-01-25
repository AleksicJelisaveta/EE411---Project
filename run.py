import torch
from datasets import load_datasets

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    permuted_train_loaders, permuted_test_loaders, rotated_train_loaders, rotated_test_loaders = load_datasets()





if __name__ == "__main__":
    main()
