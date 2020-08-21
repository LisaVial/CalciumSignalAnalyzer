import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r'G:\Lisa\data\Niko\Cortex7 10x 20fps gain 2 wash in 20 mM KCl_MMStack_Default_1-1900_normlized.csv',
                   delimiter=';' )
corr = data.corr()
print(corr)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr, cmap='PuBuGn', vmin=0, vmax=1)
fig.colorbar(cax)
tick_names = data.columns
ticks = np.arange(0, len(data.columns), 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
labels = [item.get_text() for item in ax.get_xticklabels()]
empty_string_labels = [''] * len(labels)
for idx in range(len(empty_string_labels)):
    if idx % 4 == 0:
        empty_string_labels[idx] = empty_string_labels[idx].replace('', 'Mean ' + str(ticks[idx]+1))
ax.set_xticklabels(empty_string_labels)
plt.xticks(rotation=90)
ax.set_yticklabels(empty_string_labels)
plt.show()

