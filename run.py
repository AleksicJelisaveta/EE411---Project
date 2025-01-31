import argparse
from MNIST_functions import run_experiment_2A, run_experiment_2B
from generate_datasets import load_datasets
from utils import *

def main():
    parser = argparse.ArgumentParser(description='Run experiments for reproduction.')
    parser.add_argument('--experiment', type=str, choices=['A', 'B'], required=True, help='Experiment type (A or B)')
    parser.add_argument('--task_type', type=str, choices=['permute', 'rotate'], required=True, help='Task type (permute or rotate)')
    parser.add_argument('--num_tasks', type=int, required=True, help='Number of tasks (positive integer)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    parser.add_argument('--lambda_ewc', type=float, default=10.0, help='Regularization strength for EWC (default: 1000.0)')

    args = parser.parse_args()

    num_tasks_permute = args.num_tasks if args.task_type == 'permute' else 1
    num_tasks_rotate = args.num_tasks if args.task_type == 'rotate' else 1

    # Load datasets
    permuted_train_loaders, permuted_test_loaders, rotated_train_loaders, rotated_test_loaders = load_datasets(
        num_tasks_permute=num_tasks_permute, num_tasks_rotate=num_tasks_rotate)

    if args.task_type == 'permute':
        train_loaders = permuted_train_loaders
        test_loaders = permuted_test_loaders
    elif args.task_type == 'rotate':
        train_loaders = rotated_train_loaders
        test_loaders = rotated_test_loaders

    if args.experiment == 'A':
        # additional arguments for experiment A, not needed for B
        # hidden layer width, default 400, learning rate, default 0.001, dropout_input, default 0, dropout_hidden, default 0

        parser = argparse.ArgumentParser(description='Neural Networks hyperparameters')
        parser.add_argument('--hidden_layer_width', type=int, default=400, help='Width of hidden layer (default: 400)')
        parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate (default: 0.001)')
        parser.add_argument('--dropout_input', type=float, default=0, help='Dropout rate for input layer (default: 0)')
        parser.add_argument('--dropout_hidden', type=float, default=0, help='Dropout rate for hidden layer (default: 0)')

        epoch_accuracies_SGD, epoch_accuracies_EWC, epoch_accuracies_L2 = run_experiment_2A(train_loaders, test_loaders, args.num_tasks, args.lambda_ewc, args.hidden_layer_width, args.learning_rate, args.dropout_input, args.dropout_hidden)
        save_experiment_results_2A('experiment_A_results.csv', epoch_accuracies_SGD, epoch_accuracies_EWC, epoch_accuracies_L2)

    elif args.experiment == 'B':
        avg_sgd_accuracy, avg_ewc_accuracy = run_experiment_2B(train_loaders, test_loaders, args.num_tasks, args.lambda_ewc)
        save_fraction_correct_results(avg_ewc_accuracy, avg_sgd_accuracy, args.num_tasks, 'experiment_B_results.csv')


if __name__ == '__main__':
    main()
