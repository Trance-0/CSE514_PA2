# -*- coding: utf-8 -*-
# File : dimension_reduce.py
# Time : 2024/4/23 7:07 
# Author : Dijkstra Liu
# Email : l.tingjun@wustl.edu
# 
# 　　　    /＞ —— フ
# 　　　　　| `_　 _ l
# 　 　　　ノ  ミ＿xノ
# 　　 　 /　　　 　|
# 　　　 /　 ヽ　　ﾉ
# 　 　 │　　|　|　\
# 　／￣|　　 |　|　|
#  | (￣ヽ＿_ヽ_)__)
# 　＼_つ
#
# Description:
from customer_personality_analysis.customer_personality_analysis import customer_personality_analysis
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


cpa = customer_personality_analysis()
X, y = cpa.prepared_standardize_data(data_name='married', debug=False)


print("Original shape:", X.shape)

pca = PCA(n_components=9)
X_pca = pca.fit_transform(X)
print("Reduced shape:", X_pca.shape)

X, X_final_val, y, y_final_val = train_test_split(X, y, test_size=0.2, random_state=42)
np.savetxt('X_train_reduced.csv', X, delimiter=',')
np.savetxt('X_val_reduced.csv', X_final_val, delimiter=',')
np.savetxt('y_train_reduced.csv', y, delimiter=',')
np.savetxt('y_val_reduced.csv', y_final_val, delimiter=',')
