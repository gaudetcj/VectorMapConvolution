import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


real_path = r"C:\Projects\VectorMapConvolution_2\data\satellite_data\val_log.txt"
vmap_path = r"C:\Projects\VectorMapConvolution_2\data\satellite_data\val_log_quat.txt"


real_val = pd.read_csv(real_path, names=['val'])
real_val = real_val['val'].apply(lambda x: float(x.split(':')[1][1:])).values
vmap_val = pd.read_csv(vmap_path, names=['val'])
vmap_val = vmap_val['val'].apply(lambda x: float(x.split(':')[1][1:])).values


plt.figure(figsize=(4, 3))
plt.plot(real_val)
plt.plot(vmap_val)
plt.xlabel('epochs')
plt.ylabel('Jaccard score')
plt.legend(['Real', 'Vector map'], loc=4)
plt.grid()
plt.savefig(r"C:\Projects\VectorMapConvolution_2\data\satellite_data\results.png", dpi=300, bbox_inches="tight")
