""" Data analysis class for algerian forest fires

Display the metadata for dataset

Show variable information

Show feature vector heads

Show target vector heads

"""

import os
from pathlib import Path
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# appending a path
sys.path.append(os.path.join(Path(__file__).resolve().parent.parent,'customer_personality_analysis'))
# print(Path(__file__).resolve().parent.parent)

from customer_personality_analysis import customer_personality_analysis
 
cpa_data = customer_personality_analysis().data()
df=cpa_data['data']
num_df=df[cpa_data['numerical']]

print(df.describe())

# show covariance matrix
# example code from: https://stackoverflow.com/questions/3529666/matplotlib-matshow-labels
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(abs(num_df.corr()), interpolation='nearest',cmap='plasma')
print('Correlation matrix \n',num_df.corr())

# display color bar
cb = fig.colorbar(cax)

# set title and texts
plt.title('Correlation Matrix for numerical features in customer personality analysis', fontsize=16)
alpha=list(num_df.columns)

# force display all labels
ax.set_xticks(np.arange(len(alpha)))
ax.set_yticks(np.arange(len(alpha)))
ax.set_xticklabels(alpha,rotation = 50)
ax.set_yticklabels(alpha)

fig=plt.gcf()
fig.subplots_adjust(right=0.83,top=0.71,hspace=0.12,wspace=0.18,left=0)
plt.show()

correlation_matrix = num_df.corr()
