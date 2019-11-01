import os
import pandas as pd
import matplotlib.pyplot as plt


folder_path = './results'
fig, axs = plt.subplots(nrows=1, ncols=2, dpi=100, figsize=(18, 8))
exp_names = []
for file_name in os.listdir(folder_path):
    exp_names.append(file_name.split('_')[1].split('.')[0])
    file_path = os.path.join(folder_path, file_name)
    results = pd.read_csv(file_path, header=None, names=['Loss', 'Acc'])
    
    axs[0].plot(results.index, results.Loss)
    axs[1].plot(results.index, results.Acc)

axs[0].set_xlabel('Epochs', fontsize=14)
axs[0].set_ylabel('Loss', fontsize=14)
axs[0].legend(exp_names, fontsize=12)
axs[0].tick_params(axis='both', which='major', labelsize=12)
axs[0].tick_params(axis='both', which='minor', labelsize=10)

axs[1].set_xlabel('Epochs', fontsize=14)
axs[1].set_ylabel('Acc', fontsize=14)
axs[1].legend(exp_names, fontsize=12)
axs[1].tick_params(axis='both', which='major', labelsize=12)
axs[1].tick_params(axis='both', which='minor', labelsize=10)

plt.show()