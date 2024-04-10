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

cpa = customer_personality_analysis()
X, y = cpa.prepared_standardize_data(data_name='married', debug=False)


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

        if activation == 'relu':
            activation_function = nn.ReLU()
        elif activation == 'tanh':
            activation_function = nn.Tanh()
        elif activation == 'leakyrelu':
            activation_function = nn.LeakyReLU(negative_slope=0.01)
        else:
            raise ValueError("Unsupported activation function. Choose from 'relu', 'tanh', or 'leakyrelu'.")

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

        self.activation = activation_function

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


hyperparams_space = {
    'units': [10],
    'activation': ['relu'],
    'learning_rate': [0.002],
    'epochs': [10],
    'batch_size': [32]
}

device = "cuda"
results = []

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for config, (units, activation, learning_rate, epochs, batch_size) in enumerate(product(*hyperparams_space.values())):
    fold_results = []
    print(
        f"\nConfig {config + 1}/{len(list(product(*hyperparams_space.values())))}: Units-{units}, Activation-{activation}, LearningRate-{learning_rate}, Epochs-{epochs}, BatchSize-{batch_size}")
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        print(f"  Fold {fold + 1}/5")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        train_dataset = CPADataset(X_train, y_train)
        test_dataset = CPADataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = SimpleANN(input_size=X_train.shape[1], units=units, activation=activation).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs.squeeze(), labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"    Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader)}")

        # 验证模型
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                predicted = (outputs.squeeze() > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        fold_results.append(accuracy)
        print(f"  Fold {fold + 1}/5, Accuracy: {accuracy}")

    avg_accuracy = np.mean(fold_results)
    results.append({
        'units': units,
        'activation': activation,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'batch_size': batch_size,
        'accuracy': avg_accuracy
    })
    print(f"Config {config + 1} Average Accuracy: {avg_accuracy}")

best_params = max(results, key=lambda x: x['accuracy'])
print("\nBest hyperparameters:", best_params)
