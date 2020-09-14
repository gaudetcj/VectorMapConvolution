import os
import pandas as pd
import matplotlib.pyplot as plt


folder_path = './results'
fig, axs = plt.subplots(nrows=1, ncols=2, dpi=300, figsize=(8, 3))
exp_names = []
for file_name in os.listdir(folder_path):
    exp_names.append(file_name.split('_')[1].split('.')[0])
    file_path = os.path.join(folder_path, file_name)
    results = pd.read_csv(file_path, header=None, names=['Loss', 'Acc'])

    axs[0].plot(results.index, results.Loss, linewidth=0.5)
    axs[1].plot(results.index, results.Acc, linewidth=0.5)

axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].legend(exp_names)

axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Acc')
axs[1].legend(exp_names)

plt.savefig(r"C:\Projects\VectorMapConvolution_2\data\cifar_data\results.png", dpi=300, bbox_inches="tight")