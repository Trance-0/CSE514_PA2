"""Print scatter of data"""
import os
from pathlib import Path
import sys
import matplotlib as mpl
from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# appending a path
sys.path.append(os.path.join(Path(__file__).resolve().parent.parent,'customer_personality_analysis'))
# print(Path(__file__).resolve().parent.parent)

from customer_personality_analysis import customer_personality_analysis
 
cpa_data = customer_personality_analysis().data()
df=cpa_data['data'][cpa_data['numerical']]

# data frame and render color
df=df[cpa_data['numerical']]
default_color='#4a917d'
line_color='#0a7353'
cm = mpl.colormaps.get_cmap('viridis')

# plot variables
plotSize=20
textSize=7
point_alpha=0.3
point_size=10
label_size=10
label_tick_size=8

# predictor variables, leave it empty for running default colors
predictor_name=''
y_unit=''
y_label_rotate=45

plot_title='Visualization of numerical value in Customer Personality Analysis'
# desired distance of legend label
plot_legend_unit=10

### config ends ###

# code reference:https://www.kaggle.com/code/djsquiggle/starter-red-wine-quality-be9850f8-e
df = df.select_dtypes(include =[np.number]) # keep only numerical columns

# Remove rows and columns that would lead to df being singular
df = df.dropna()

# keep columns where there are more than 1 unique values
df = df[[col for col in df if df[col].nunique() > 1]]

y=np.ravel(df.loc[:, df.columns==predictor_name].to_numpy())
columnNames = list(df)
norm_min=0 if predictor_name=='' else y.min()
norm_max=0 if predictor_name=='' else y.max()

# reduce the number of columns for matrix inversion of kernel density plots
if len(columnNames) > 10:
    columnNames = columnNames[:10]
df = df[columnNames]

ax = pd.plotting.scatter_matrix(df, c=default_color if predictor_name=='' else y, alpha=point_alpha, figsize=[plotSize, plotSize], diagonal='kde',s=point_size,density_kwds={'color':line_color})

# set plot label size
for item in ax.ravel():
    # tick lable
    plt.setp(item.xaxis.get_majorticklabels(), 'size', label_tick_size) 
    plt.setp(item.yaxis.get_majorticklabels(), 'size', label_tick_size)
    # label
    plt.setp(item.xaxis.get_label(), 'size', label_size) 
    plt.setp(item.yaxis.get_label(), 'size', label_size) 
    plt.setp(item.yaxis.get_label(), 'rotation', y_label_rotate) 

corrs = df.corr().values
for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
    ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j],
                      (0, 0.9),
                      xycoords='axes fraction',
                      ha='left',
                      va='center',
                      size=textSize)
plt.suptitle(plot_title)

if predictor_name!='':
    # load legend
    lhandles=[]
    norm = mpl.colors.Normalize(vmin=norm_min, vmax=norm_max)

    for i in range(int(norm_min),int(norm_max),plot_legend_unit):
        lhandles.append(patches.Patch(color=cm(norm(i)), label=f'{i} {y_unit}'))
    # Put a legend to the right of the current axis
    plt.legend(bbox_to_anchor=(1.02, 1), loc='center left',handles=lhandles)

fig=plt.gcf()
fig.subplots_adjust(top=0.93)
plt.show()
