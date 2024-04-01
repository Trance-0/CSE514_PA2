import os
from pathlib import Path
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

# appending a path
sys.path.append(os.path.join(Path(__file__).resolve().parent.parent,'customer_personality_analysis'))
# print(Path(__file__).resolve().parent.parent)
from customer_personality_analysis import customer_personality_analysis
 
cpa_data = customer_personality_analysis().married_data()
df=cpa_data['data']

# config
predictor_name='Income_level'
test_size=0.1
seed=347573952
n_neighbors=[3,5,9,17,21,33]

X=df.loc[:, df.columns != predictor_name]
y=df.loc[:, df.columns == predictor_name]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

print(type(X_train))
scores=[]

for neighbor in n_neighbors:
    kf = KFold(n_splits=5,shuffle=True,random_state=seed)
    for fold,(train_idx, validate_idx) in enumerate(kf.split(X_train,y_train)):
        print(train_idx[:5],validate_idx[:5])
        cur_X_train=[X_train[i] for i in train_idx]
        cur_X_validate=[X_train[i] for i in validate_idx]
        cur_y_train=[y_train[i] for i in train_idx]
        cur_y_validate =[y_train[i] for i in validate_idx]
        knn=KNeighborsClassifier(n_neighbors=neighbor,n_jobs=-1)
        knn.fit(cur_X_train, cur_y_train)
        scores.append((fold,neighbor,knn.score(cur_X_validate, cur_y_validate)))

print(scores)