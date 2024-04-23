# -*- coding: utf-8 -*-
# File : svm_model.py
# Time : 2024/4/10 10:22
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
# This is the ANN model, using pytorch to test all hyperparameter and use 5-Fold to test them.
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import numpy as np
from itertools import product

from customer_personality_analysis.customer_personality_analysis import customer_personality_analysis
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import numpy as np
from itertools import product
import time

# Assuming cpa.prepared_standardize_data() appropriately retrieves and preprocesses the data
cpa = customer_personality_analysis()
# X, y = cpa.prepared_standardize_data(data_name='married', debug=False)
X, y = cpa.prepare_reduced_data(data_name='single')

# Split final validation set
from sklearn.model_selection import train_test_split
X, X_final_val, y, y_final_val = train_test_split(X, y, test_size=0.2, random_state=42)
# np.savetxt('X_train.csv', X, delimiter=',')
# np.savetxt('X_val.csv', X_final_val, delimiter=',')
# np.savetxt('y_train.csv', y, delimiter=',')
# np.savetxt('y_val.csv', y_final_val, delimiter=',')

class CPADataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class SimpleANN(nn.Module):
    def __init__(self, input_size, units, activation):
        super(SimpleANN, self).__init__()
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'leakyrelu': nn.LeakyReLU(negative_slope=0.01)
        }
        self.fc1 = nn.Linear(input_size, units)
        self.bn1 = nn.BatchNorm1d(units)
        self.fc2 = nn.Linear(units, 2 * units)
        self.bn2 = nn.BatchNorm1d(2 * units)
        self.fc3 = nn.Linear(2 * units, 4 * units)
        self.bn3 = nn.BatchNorm1d(4 * units)
        self.dropout1 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(4 * units, 2 * units)
        self.bn4 = nn.BatchNorm1d(2 * units)
        self.fc5 = nn.Linear(2 * units, 1)
        self.output = nn.Sigmoid()
        self.activation = activations[activation]

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.bn2(x)
        x = self.fc3(x)
        x = self.activation(x)
        x = self.bn3(x)
        x = self.dropout1(x)
        x = self.fc4(x)
        x = self.activation(x)
        x = self.bn4(x)
        x = self.fc5(x)
        x = self.output(x)
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"
hyperparams_space = {
    'units': [5, 10],
    'activation': ['relu', 'tanh'],
    'learning_rate': [0.1, 0.01],
    'epochs': [20],
    'batch_size': [16]
}
kf = KFold(n_splits=5, shuffle=True, random_state=42)
all_results = []

for config, (units, activation, learning_rate, epochs, batch_size) in enumerate(product(*hyperparams_space.values())):
    fold_results = []
    print(f"\nConfig {config + 1}: Units-{units}, Activation-{activation}, LearningRate-{learning_rate}, Epochs-{epochs}, BatchSize-{batch_size}")
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = y[train_index], y[test_index]

        train_dataset = CPADataset(X_train, y_train)
        val_dataset = CPADataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = SimpleANN(input_size=X_train.shape[1], units=units, activation=activation).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Timing the training process
        start_time = time.time()
        model.train()
        for epoch in range(epochs):
            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs.squeeze(), labels)
                loss.backward()
                optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            correct, total = 0, 0
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                predicted = (outputs.squeeze() > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        fold_results.append(accuracy)
        print(f"  Fold {fold + 1}/5 Accuracy: {accuracy:.4f}")
    fold_time = time.time() - start_time

    avg_accuracy = np.mean(fold_results)
    all_results.append({
        'units': units,
        'activation': activation,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'batch_size': batch_size,
        'accuracy': avg_accuracy,
        'time': fold_time
    })
    print(f"Average Accuracy: {avg_accuracy:.4f}, Time: {fold_time:.2f} seconds")
# Finding the best configuration
best_params = max(all_results, key=lambda x: x['accuracy'])
print("\nBest hyperparameters:", best_params)

# Preparing full training set and final validation set
full_train_dataset = CPADataset(X, y)
final_val_dataset = CPADataset(X_final_val, y_final_val)

full_train_loader = DataLoader(full_train_dataset, batch_size=best_params['batch_size'], shuffle=True)
final_val_loader = DataLoader(final_val_dataset, batch_size=best_params['batch_size'], shuffle=False)

# Model training with best hyperparameters
final_model = SimpleANN(input_size=X.shape[1], units=best_params['units'], activation=best_params['activation']).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(final_model.parameters(), lr=best_params['learning_rate'])

start_time = time.time()
final_model.train()
for epoch in range(best_params['epochs']):
    for features, labels in full_train_loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = final_model(features)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()

# Evaluation on the final validation set
final_model.eval()
with torch.no_grad():
    correct, total = 0, 0
    for features, labels in final_val_loader:
        features, labels = features.to(device), labels.to(device)
        outputs = final_model(features)
        predicted = (outputs.squeeze() > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
final_accuracy = correct / total
final_time = time.time() - start_time

print(f"\nFinal Model Performance: Accuracy - {final_accuracy:.4f}")
print(f"Training and Evaluation Time: {final_time:.2f} seconds")

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
x_ticks = [f"Config {i+1}" for i in range(len(all_results))]
accuracies = [result['accuracy'] for result in all_results]
ax.bar(x_ticks, accuracies, color='blue')
ax.set_xlabel('Configurations')
ax.set_ylabel('Average Accuracy')
ax.set_title('5-Fold Cross-Validation Results')
plt.xticks(rotation=45)
plt.show()

