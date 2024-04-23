import os
from pathlib import Path
import sys
import matplotlib as mpl
from matplotlib import patches
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# appending a path
sys.path.append(os.path.join(Path(__file__).resolve().parent.parent, 'customer_personality_analysis'))
# print(Path(__file__).resolve().parent.parent)
from customer_personality_analysis import customer_personality_analysis

cpa = customer_personality_analysis()
population = cpa.married_data()['data']
print(population.head())
cpa_data = cpa.numerical_encode(population, bisected_data=True)
df = cpa_data['data']
print(df.head())

# config
predictor_name = 'Income_level'
test_size = 0.1
seed = 347573952
n_neighbors = [3, 5, 9, 17, 21, 33]
debug = True
cm = mpl.colormaps.get_cmap('viridis')
fold_size = 5

X = df.loc[:, df.columns != predictor_name].to_numpy()
y = df.loc[:, df.columns == predictor_name].to_numpy().ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

if debug: print(type(X_train))
scores = []

for neighbor in n_neighbors:
    kf = KFold(n_splits=fold_size, shuffle=True, random_state=seed)
    for fold, (train_idx, validate_idx) in enumerate(kf.split(X_train, y_train)):
        if debug: print(train_idx[:5], validate_idx[:5])
        cur_X_train = [X_train[i] for i in train_idx]
        cur_X_validate = [X_train[i] for i in validate_idx]
        cur_y_train = [y_train[i] for i in train_idx]
        cur_y_validate = [y_train[i] for i in validate_idx]
        knn = KNeighborsClassifier(n_neighbors=neighbor, n_jobs=-1)
        knn.fit(cur_X_train, cur_y_train)
        scores.append((fold, neighbor, knn.score(cur_X_validate, cur_y_validate)))

if debug: print(scores)

norm = mpl.colors.Normalize(vmin=min(n_neighbors), vmax=max(n_neighbors))

plt.title("Folding result for each neighbor count")
for i in range(len(n_neighbors)):
    x = range(0, fold_size)
    y = [scores[idx][2] for idx in range(i * fold_size, (i + 1) * fold_size)]
    plt.plot(x, y, c=cm(norm(n_neighbors[i])))
    plt.xlabel('K-fold trial')
    plt.ylabel('Accuracy')

ax = plt.gca()
ax.set_ylim([0, 1])
# load legend
lhandles = []

for i in range(len(n_neighbors)):
    lhandles.append(patches.Patch(color=cm(norm(n_neighbors[i])), label=f'neighbor count: {n_neighbors[i]}'))
# Put a legend to the right of the current axis
plt.legend(bbox_to_anchor=(1, 0), loc="lower right", handles=lhandles)

plt.show()
