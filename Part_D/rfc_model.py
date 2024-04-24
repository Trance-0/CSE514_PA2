"""
with the same testing as Dijkstra
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report

# import in vscode
import os
from pathlib import Path
import sys
# appending a path
sys.path.append(os.path.join(Path(__file__).resolve().parent.parent,'customer_personality_analysis'))
# print(Path(__file__).resolve().parent.parent)
from customer_personality_analysis import customer_personality_analysis

# import in idea
# from customer_personality_analysis.customer_personality_analysis import customer_personality_analysis

# Initialize customer personality analysis
cpa = customer_personality_analysis()

# Prepare and reduce the data
# X, y = cpa.prepare_reduced_data(data_name="single")
# X, y = cpa.prepare_reduced_data(data_name="partner_loss")
X, y = cpa.prepare_reduced_data(data_name="married")

SEED=32

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)

# Initialize the StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the model and the grid of hyperparameters to search over
model = RandomForestClassifier(n_jobs=-1)
param_grid = {
    'n_estimators':[1,2,4,8,16,32,64,128,256,512,1024],
    'max_samples':[0.1,0.3,0.5,0.7,0.9]
}

# Perform grid search with 5-fold cross-validation
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
start_time = time.time()
grid_search.fit(X_train_scaled, y_train)
training_time = time.time() - start_time

# Display the best hyperparameters and the corresponding cross-validation score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# Visualize the cross-validation results
cv_results = grid_search.cv_results_
scores_mean = cv_results['mean_test_score']
scores_std = cv_results['std_test_score']
params = [str(x) for x in cv_results['params']]

x = np.arange(len(scores_mean))
plt.errorbar(x, scores_mean, yerr=scores_std, fmt='o')
plt.xticks(x, params, rotation=45, ha='right')
plt.xlabel('Parameter Set')
plt.ylabel('Mean CV Score')
plt.title('Cross Validation Results')
plt.tight_layout()
plt.show()

# Train the best model on the whole training set
best_model = grid_search.best_estimator_
start_time = time.time()
best_model.fit(X_train_scaled, y_train)
retraining_time = time.time() - start_time

# Predict on the final validation set
y_pred = best_model.predict(X_test_scaled)
prediction_time = time.time() - start_time

# Report the performance
print("Classification report on test data:")
print(classification_report(y_test, y_pred))
print(f"Training time: {retraining_time:.4f} seconds")
print(f"Prediction time: {prediction_time:.4f} seconds")
