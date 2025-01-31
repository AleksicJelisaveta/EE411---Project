import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv

# Extracting accuracies from csv file
def extract_epoch_accuracies(file_path):
    df = pd.read_csv(file_path)
    
    tasks = sorted(df['Train_Task'].unique())
    
    epoch_accuracies_SGD = {task: [] for task in tasks}
    epoch_accuracies_EWC = {task: [] for task in tasks}
    epoch_accuracies_L2 = {task: [] for task in tasks}
    
    for task in tasks:
        for val_task in tasks:
            subset = df[(df['Train_Task'] == task) & (df['Eval_Task'] == val_task)]
            
            epoch_accuracies_SGD[task].append(list(subset['SGD_Accuracy']))
            epoch_accuracies_EWC[task].append(list(subset['EWC_Accuracy']))
            epoch_accuracies_L2[task].append(list(subset['L2_Accuracy']))
    
    return epoch_accuracies_SGD, epoch_accuracies_EWC, epoch_accuracies_L2
    

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def plot_experiment_2A(epoch_accuracies_SGD, epoch_accuracies_EWC, epoch_accuracies_L2, title):
    def flatten(lst):
        return [item for sublist in lst for item in sublist]
    
    sgd_data, ewc_data, l2_data, time_data = [], [], [], []
    
    # Collect all accuracy values to determine y-axis limits
    all_accuracies = []

    for task_index in range(len(epoch_accuracies_SGD)):
        sgd_task = [epoch_accuracies_SGD[i][task_index] for i in range(len(epoch_accuracies_SGD[task_index]))]
        ewc_task = [epoch_accuracies_EWC[i][task_index] for i in range(len(epoch_accuracies_EWC[task_index]))]
        l2_task = [epoch_accuracies_L2[i][task_index] for i in range(len(epoch_accuracies_L2[task_index]))]

        sgd_flat, ewc_flat, l2_flat = flatten(sgd_task), flatten(ewc_task), flatten(l2_task)

        sgd_data.append(sgd_flat)
        ewc_data.append(ewc_flat)
        l2_data.append(l2_flat)

        all_accuracies.extend(sgd_flat + ewc_flat + l2_flat)

        time_data.append(np.arange(len(sgd_flat)))

    # Define consistent y-axis limits
    y_min, y_max = min(all_accuracies), max(all_accuracies)
    y_max = y_max + 0.1 * (y_max - y_min)


    figure_height = len(sgd_data)
    fig, axes = plt.subplots(len(sgd_data), 1, figsize=(10, figure_height), sharex=True)
    colors = {'ewc': 'red', 'l2': 'green', 'sgd': 'blue'}
    max_len = len(sgd_data[0])
   
    # Add title to the first subplot
    axes[0].set_title(title, fontsize=14, loc='center', pad=10)

    for task_index in range(len(sgd_data)):
        time = time_data[0]
        
        def pad_with_nans(original, max_len):
            return [np.nan] * (max_len - len(original)) + original

        sgd_padded = pad_with_nans(sgd_data[task_index], max_len)
        ewc_padded = pad_with_nans(ewc_data[task_index], max_len)
        l2_padded = pad_with_nans(l2_data[task_index], max_len)

        # Plot each task
        axes[task_index].plot(time, sgd_padded, label="SGD", color=colors['sgd'], alpha=0.8)
        axes[task_index].plot(time, ewc_padded, label="EWC", color=colors['ewc'], alpha=0.8)
        axes[task_index].plot(time, l2_padded, label="L2", color=colors['l2'], alpha=0.8)

        # Vertical dashed lines
        for i in range(19, max_len, 20):
            axes[task_index].axvline(x=i, color='gray', linestyle='--', linewidth=1)

        axes[task_index].set_ylabel(f"Task {chr(65 + task_index)}")
        axes[task_index].set_ylim(y_min, y_max)  # Apply fixed y-axis limits
        axes[task_index].legend(loc="lower left", fontsize='xx-small')
        axes[task_index].set_xticks([])
        if task_index < len(sgd_data) - 1:
            axes[task_index].set_xticklabels([])

    axes[-1].set_xlabel("Training time")
    axes[-1].text(-0.1, -0.3, "Fraction correct", transform=axes[-1].transAxes, fontsize=10, ha="left", va="bottom")
  
    plt.tight_layout()
    plt.show()


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

def save_experiment_results_2A(filename, epoch_accuracies_SGD, epoch_accuracies_EWC, epoch_accuracies_L2):
    
    rows = []
    max_epochs=20
    # Loop through training tasks
    for train_task in epoch_accuracies_SGD:
        # Loop through evaluation tasks
        for eval_task in epoch_accuracies_SGD[train_task]:
            for epoch in range(max_epochs):
                rows.append({
                    'Epoch': epoch + 1,  # Epochs start at 1
                    'Train_Task': train_task,
                    'Eval_Task': eval_task,
                    'SGD_Accuracy': epoch_accuracies_SGD[train_task][eval_task][epoch],
                    'EWC_Accuracy': epoch_accuracies_EWC[train_task][eval_task][epoch],
                    'L2_Accuracy': epoch_accuracies_L2[train_task][eval_task][epoch],
                })

    # Save to CSV
    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['Epoch', 'Train_Task', 'Eval_Task', 'SGD_Accuracy', 'EWC_Accuracy', 'L2_Accuracy'])
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Data saved to {filename}")    