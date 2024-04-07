"""
random forest classifier for populations
"""
import math
import os
from pathlib import Path
import sys
import matplotlib as mpl
from matplotlib import patches
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# appending a path
sys.path.append(os.path.join(Path(__file__).resolve().parent.parent,'customer_personality_analysis'))
# print(Path(__file__).resolve().parent.parent)
from customer_personality_analysis import customer_personality_analysis
 
cpa = customer_personality_analysis()

# config
predictor_name='Income_level'
test_size=0.1
seed=347573952
n_estimators=[1,2,4,8,16,32,64,128,256,512,1024]
debug=True
cm = mpl.colormaps.get_cmap('viridis')
fold_size=5
populations={"married population":cpa.married_data()['data'],
            "single population":cpa.single_data()['data'],
            "partner lost population":cpa.partner_loss_data()['data'],
             }
debug=False

for name, population in populations.items():
    if debug: print(population.head())
    cpa_data=cpa.numerical_encode(population,bisected_data=True)
    df=cpa_data['data']
    if debug: print(df.head())

    X=df.loc[:, df.columns != predictor_name].to_numpy()
    y=df.loc[:, df.columns == predictor_name].to_numpy().ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    if debug: print(type(X_train))
    scores=[]
    average_performance=[]

    for estimator in n_estimators:
        cur_total_performance=0
        kf = KFold(n_splits=fold_size,shuffle=True,random_state=seed)
        for fold,(train_idx, validate_idx) in enumerate(kf.split(X_train,y_train)):
            if debug: print(train_idx[:5],validate_idx[:5])
            cur_X_train=[X_train[i] for i in train_idx]
            cur_X_validate=[X_train[i] for i in validate_idx]
            cur_y_train=[y_train[i] for i in train_idx]
            cur_y_validate =[y_train[i] for i in validate_idx]
            rfc = RandomForestClassifier(n_estimators=estimator, criterion='gini',n_jobs=-1,random_state=seed)
            rfc.fit(cur_X_train, cur_y_train)
            cur_recall=recall_score(cur_y_validate,rfc.predict(cur_X_validate))
            scores.append((fold,estimator,cur_recall))
            cur_total_performance+=cur_recall
        average_performance.append((estimator,cur_total_performance/fold_size))
        if debug: print(scores)

        norm = mpl.colors.Normalize(vmin=math.log2(min(n_estimators)), vmax=math.log2(max(n_estimators)))

        plt.title(f"Folding result for each estimator count {estimator} in {name}")
        for i in range(len(n_estimators)):
            x=range(0,fold_size)
            # read latest data
            y=[scores[idx][-1] for idx in range(len(scores)-fold_size,len(scores))]
            plt.plot(x,y,c=cm(norm(math.log2(n_estimators[i]))))
            plt.xlabel('K-fold trial')
            plt.ylabel('Recall')

        ax = plt.gca()
        ax.set_ylim([0.5, 1])
        # load legend
        lhandles=[]

        for i in range(len(n_estimators)):
            lhandles.append(patches.Patch(color=cm(norm(n_estimators[i])), label=f'sample proportion: {n_estimators[i]}'))
        # Put a legend to the right of the current axis
        plt.legend(bbox_to_anchor=(1,0), loc="lower right",handles=lhandles)

        plt.show()

    best_performance_pair=max(enumerate(average_performance), key=lambda x:x[-1])
    print(f'best set of parameter for population {name} is {[best_performance_pair[0]]} with average recall {best_performance_pair[1]}')


