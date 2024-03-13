#
# Rollings, A. (2020). mlrose: Machine Learning, Randomized Optimization and Search package for
# Python, hiive extended remix. https://github.com/hiive/mlrose/blob/master/problem_examples.ipynb
#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
from torch import nn, optim
from torch.nn import functional as F
import mlrose_hiive as mlrose
import time

def get_accuracy(x_train_nn, y_train_nn):
    with torch.no_grad():
        no_samples=0
        no_preds=0
        tp=0
        tn=0
        fp=0
        fn=0
        for i in range(0, list(y_train_nn.size())[0]):
            prediction = net(x_train_nn[i])
            predicted_class = np.argmax(prediction)
            no_samples=no_samples+1
            if(predicted_class==y_train_nn[i]):
                no_preds=no_preds+1
                if(predicted_class==1):
                    tp=tp+1
                else:
                    tn=tn+1
            else:
                if(predicted_class==1):
                    fp=fp+1
                else:
                    fn=fn+1
        if no_samples == 0:
            accuracy = 0
        else:
            accuracy = no_preds / no_samples

        if (tp+fp) == 0:
            precision = 0
        else:
            precision = tp / (tp + fp)

        if (tp+fn) == 0:
            recall = 0
        else:
            recall=tp/(tp+fn)

        if (precision+recall) == 0:
            f1 = 0
        else:
            f1=2*(precision*recall)/(precision+recall)
        return {'accuracy':accuracy,'precision':precision,'recall':recall,'f1':f1}

bank=pd.read_csv('bank.csv')

bank.deposit.value_counts()

bank['deposit'][bank['deposit'] == 'yes'] =1
bank['deposit'][bank['deposit'] == 'no'] = 0
bank['deposit']=bank['deposit'].astype(int)
bank2 = pd.get_dummies(bank, drop_first=True)
bank2_norm = (bank2.drop('deposit',1) - np.min(bank2.drop('deposit',1))) / (np.max(bank2.drop('deposit',1)) - np.min(bank2.drop('deposit',1))).values
bank2_norm['deposit']=bank2.deposit
train_norm, test_norm = train_test_split(bank2_norm, test_size = 0.3,random_state=0)

train_X_norm = train_norm.drop('deposit',1)
train_y_norm=train_norm.deposit
test_X_norm= test_norm.drop('deposit',1)
test_y_norm =test_norm.deposit

x_train_nn = torch.tensor(train_X_norm.to_numpy()).float()
x_test_nn = torch.tensor(test_X_norm.to_numpy()).float()
y_train_nn = torch.tensor(train_y_norm.values).long()
y_test_nn = torch.tensor(test_y_norm.values).long()


class bankNN(nn.Module):
    def __init__(self):
        super(bankNN, self).__init__()
        self.fc1 = nn.Linear(42, 500)
        self.fc2 = nn.Linear(500, 350)
        self.fc3 = nn.Linear(350, 200)
        self.fc5 = nn.Linear(200, 100)
        self.fc6 = nn.Linear(100, 20)
        self.fc4 = nn.Linear(20, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        return self.fc4(x)

for i in [0.1,0.2,0.3,0.4]:
    # print('i',i)
    net = bankNN()
    optimizer = optim.SGD(net.parameters(),lr=i)
    criterion = nn.CrossEntropyLoss()
    losses = []
    for epoch in range(1, 200):
        # print('epoch', epoch)
        optimizer.zero_grad()
        outputs = net(x_train_nn)
        loss = criterion(outputs, y_train_nn)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    plt.plot(losses,label="Learning rate: "+str(i))
    plt.legend()
    fig=plt.gcf()
fig.set_size_inches(10,6)
plt.xlabel("Epoch", fontsize=18)
plt.ylabel("Loss", fontsize=18)
plt.show()

for i in [0.001,0.01,0.1]:
    # print('i',i)
    net = bankNN()
    optimizer = optim.SGD(net.parameters(),lr=0.3,momentum=i)
    criterion = nn.CrossEntropyLoss()
    losses = []
    for epoch in range(1, 500):
        # print('epoch', epoch)
        optimizer.zero_grad()
        outputs = net(x_train_nn)
        loss = criterion(outputs, y_train_nn)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    plt.plot(losses,label="Momentum: "+str(i))
    plt.legend()
    fig=plt.gcf()
fig.set_size_inches(10,6)
plt.xlabel("Epoch", fontsize=18)
plt.ylabel("Loss", fontsize=18)
plt.show()

net = bankNN()
optimizer = optim.SGD(net.parameters(),lr=0.3,momentum=0.1)
criterion = nn.CrossEntropyLoss()
losses = []
start_time=time.time()
for epoch in range(1, 2000):
    optimizer.zero_grad()
    outputs = net(x_train_nn)
    loss = criterion(outputs, y_train_nn)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    # plt.legend()
    # fig=plt.gcf()
end_time=time.time()
print("Training Accuracy:", get_accuracy(x_train_nn, y_train_nn))
print("Testing Accuracy:", get_accuracy(x_test_nn, y_test_nn))


# RHC
for i in [10,25,50]:
    print(i)
    clf_hill = mlrose.NeuralNetwork(hidden_nodes = [500,350,200,100,20], activation = 'relu',
                                algorithm = 'random_hill_climb',
                                max_iters=1000, bias = True, is_classifier = True,
                                learning_rate = 0.2, early_stopping = True, clip_max = 1e+10,
                                max_attempts = 100,curve=True)
    time_start = time.time()
    clf_hill.fit(x_train_nn, y_train_nn)
    fit_time = time.time()
    print(f'fit_time = {fit_time-time_start}')
    print(classification_report(y_test_nn, clf_hill.predict(x_test_nn)))
    print(classification_report(y_train_nn, clf_hill.predict(x_train_nn)))
    plt.plot(clf_hill.fitness_curve[:,0],label="Step Size: "+str(i))
    plt.legend()
    fig=plt.gcf()
fig.set_size_inches(10,6)
plt.xlabel("Epoch", fontsize=18)
plt.ylabel("Loss", fontsize=18)


# SA
for i in [5,10,25]:
    print(i)
    clf_hill = mlrose.NeuralNetwork(hidden_nodes = [500,350,200,100,20], activation = 'relu',
                                algorithm = 'simulated_annealing',
                                max_iters=1000, bias = True, is_classifier = True,
                                learning_rate = i, early_stopping = True, clip_max =1e+10,
                                max_attempts = 100,curve=True)
    time_start = time.time()
    clf_hill.fit(x_train_nn, y_train_nn)
    fit_time = time.time()
    print(f'fit_time = {fit_time-time_start}')
    print(classification_report(y_test_nn, clf_hill.predict(x_test_nn)))
    print(classification_report(y_train_nn, clf_hill.predict(x_train_nn)))
    plt.plot(clf_hill.fitness_curve[:,0],label="Step Size: "+str(i))
    plt.legend()
    fig=plt.gcf()
fig.set_size_inches(10,6)
plt.xlabel("Epoch", fontsize=18)
plt.ylabel("Loss", fontsize=18)


plt.show()

for i in [25]:
    print(i)
    clf_hill = mlrose.NeuralNetwork(hidden_nodes = [500,350,200,100,20], activation = 'relu',
                                algorithm = 'genetic_alg',
                                max_iters=1000, bias = True, is_classifier = True,
                                learning_rate = i, early_stopping = True, clip_max = 1e+10,
                                max_attempts = 10,curve=True,mutation_prob=0.4)
    time_start = time.time()
    clf_hill.fit(x_train_nn, y_train_nn)
    fit_time = time.time()
    print(f'fit_time = {fit_time-time_start}')
    print(classification_report(y_test_nn, clf_hill.predict(x_test_nn)))
    print(classification_report(y_train_nn, clf_hill.predict(x_train_nn)))
    plt.plot(clf_hill.fitness_curve[:,0],label="Step Size: "+str(i))
    plt.legend()
    fig=plt.gcf()
fig.set_size_inches(10,6)
plt.xlabel("Epoch", fontsize=18)
plt.ylabel("Loss", fontsize=18)
plt.show()