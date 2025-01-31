import matplotlib.pyplot as plt
import pandas as pd

def get_fraction_correct_results(ewc_file, sgd_file):

    # Load the data
    results_ewc = pd.read_csv(ewc_file, index_col=0)
    results_sgd = pd.read_csv(sgd_file, index_col=0)

    # store the last key
    last_key_ewc = results_ewc.keys()[-1]
    last_key_sgd = results_sgd.keys()[-1]

    # Find the rows with the highest accuracy for the last task
    ewc_last_task_max = results_ewc[last_key_ewc].max()
    sgd_last_task_max = results_sgd[last_key_sgd].max()

    # store the corresponding rows in a vector
    ewc_last_task_max_row = results_ewc.loc[results_ewc[last_key_ewc] == ewc_last_task_max]
    sgd_last_task_max_row = results_sgd.loc[results_sgd[last_key_sgd] == sgd_last_task_max]

    # store the accuracies for each task in a vector
    results_ewc = ewc_last_task_max_row.iloc[0, 2:].values
    results_sgd = sgd_last_task_max_row.iloc[0, 2:].values

    return results_ewc, results_sgd


def plot_fraction_correct_results(results_ewc, results_sgd, num_tasks, title):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_tasks + 1), results_sgd, label='SGD+dropout', marker='o', color='blue')
    plt.plot(range(1, num_tasks + 1), results_ewc, label='EWC', marker='o', color='red')
    plt.axhline(y=1.0, linestyle='--', color='black', label='Single Task Performance')
    plt.xlabel('Number of Tasks')
    plt.ylabel('Fraction Correct')
    plt.legend()
    plt.title(title)
    plt.grid(True)
    plt.show()

def save_fraction_correct_results(results_ewc, results_sgd, num_tasks, title):
    results_2B = pd.DataFrame({
        'Task': [f'Task {i+1}' for i in range(num_tasks)],
        'SGD Accuracy': results_sgd,
        'EWC Accuracy': results_ewc
    })

    results_2B.to_csv(title, index=False)

    print("Results saved to file:", title)