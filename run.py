import argparse
from MNIST_functions import run_experiment_2A, run_experiment_2B
from generate_datasets import load_datasets
from utils import *

def main():
    """
    Main run script to execute experiments for generating figures 2A and 2B from the original paper.
    Functions:
        main(): Parses command-line arguments and runs the specified experiment with the given parameters.
    """
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
    
    title = ""

    if args.experiment == 'A':
        
        if args.task_type == 'permute':
            title = "Training curves for random permutations"
        elif args.task_type == 'rotate':
            title = "Training curves for random rotations"

        epoch_accuracies_SGD, epoch_accuracies_EWC, epoch_accuracies_L2 = run_experiment_2A(train_loaders, test_loaders, args.num_tasks)
        save_experiment_results_2A('experiment_A_results.csv', epoch_accuracies_SGD, epoch_accuracies_EWC, epoch_accuracies_L2)
        epoch_accuracies_SGD, epoch_accuracies_EWC, epoch_accuracies_L2 = extract_epoch_accuracies('experiment_A_results.csv')
        plot_experiment_2A(epoch_accuracies_SGD, epoch_accuracies_EWC, epoch_accuracies_L2, title)

    elif args.experiment == 'B':

        if args.task_type == 'permute':
            title = "Fraction Correct vs Number of Tasks for PermutatatedMNIST"
        elif args.task_type == 'rotate':
            title = "Fraction Correct vs Number of Tasks for RotatedMNIST"

        avg_sgd_accuracy, avg_ewc_accuracy = run_experiment_2B(train_loaders, test_loaders, args.num_tasks, args.lambda_ewc)
        save_fraction_correct_results(avg_ewc_accuracy, avg_sgd_accuracy, args.num_tasks, 'experiment_B_results.csv')
        plot_fraction_correct_results(avg_ewc_accuracy, avg_sgd_accuracy, args.num_tasks, title)

if __name__ == '__main__':
    main()
