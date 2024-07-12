import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split

import matplotlib as mpl

import losslandscape as land

class TransformerEncoder(nn.Module):
    def __init__(self, features, classes):
        super(TransformerEncoder, self).__init__()
        self.linear1 = nn.Linear(features, 16)
        self.encoder1 = nn.TransformerEncoderLayer(16, 4, 16)
        self.linear2 = nn.Linear(16, 8)
        self.encoder2 = nn.TransformerEncoderLayer(8, 4, 16)
        self.linear3 = nn.Linear(8, classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.encoder1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.encoder2(x))
        x = self.linear3(x)
        return x
    
class Model(nn.Module):
    def __init__(self, features, classes, activ = nn.Tanh(), hidden = 2):
        super(Model, self).__init__()
        self.W1 = nn.Linear(features, hidden)
        self.W2 = nn.Linear(hidden, hidden)
        self.W3 = nn.Linear(hidden, classes)
        self.activ = activ
        
    def forward(self, x):
        x = self.activ(self.W1(x))
        x = self.activ(self.W2(x))
        x = self.W3(x)
        return x
    
if __name__ == '__main__':
    wine = load_wine()

    X_train, X_test, Y_train, Y_test = train_test_split(wine['data'], wine['target'], shuffle=True, test_size=0.1)

    X_train = torch.from_numpy(X_train).float()
    Y_train = torch.from_numpy(Y_train)
    X_test = torch.from_numpy(X_test).float()
    Y_test = torch.from_numpy(Y_test)

    train_labels = torch.zeros((X_train.shape[0], 3), dtype=torch.float32)
    for i in range(len(Y_train)):
        train_labels[i, Y_train[i]] = 1.0

    test_labels = torch.zeros((X_test.shape[0], 3), dtype=torch.float32)
    for i in range(len(Y_test)):
        test_labels[i, Y_test[i]] = 1.0

    model = TransformerEncoder(features=X_test.shape[1], classes=3)
    # model = Model(
    #     features = X_test.shape[1], 
    #     classes = 3, 
    #     hidden = 32,
    #     activ = nn.GELU()
    #     )

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    epochs = 50

    params = [land.ParamList(land.get_params(model))]
    losses = torch.zeros(epochs)

    for epoch in range(epochs):
        preds = model(X_train)
        loss = criterion(preds, train_labels)

        params.append(land.ParamList(land.get_params(model)))
        losses[epoch] += loss.item()

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

    model.eval()
    predictions = torch.argmax(model(X_test), dim=-1)
    print(classification_report(Y_test, predictions, target_names=wine['target_names']))
    print(f'First loss: {losses[0]:.5f}, last loss: {losses[-1]:.5f}\n')

    theta0 = params[0]
    for i in range(1, len(params)):
        theta0 = theta0 + params[i]
    theta0 = theta0 / len(params)
    
    theta0 = params[-1]

    # land.set_params(model, theta0.params)
    
    diff = (params[-1] - params[0]).norm().item()
    d1 = land.ParamList(land.get_params(model, random=True))
    d2 = land.ParamList(land.get_params(model, random=True))
    d1 = d1 / d1.norm()
    d2 = d2 / d2.norm()
    vecs = [d1, d2]

    loss_landscape = land.LossLandscapePlotting(
        model=model,
        criterion=criterion,
        device='cpu',
        data=(X_train, train_labels),
        parameters_history=params[0] + params[-1],
        loss_history=losses,
        theta0=theta0,
        vecs = vecs
    )

    trace = loss_landscape.compute_trace(every_ith=1)
    ralpha, rbeta, surface = loss_landscape.compute_landscape(trace, grid_density=25, coef=10)
    
    colorlist = [
        '#A27A52',
        '#FAEDCD',
        '#CCD5AE'
        ]
    mypalette = mpl.colors.LinearSegmentedColormap.from_list('mypalette', colorlist)
    
    loss_landscape.plot(
        trace=trace, 
        ralpha=ralpha, rbeta=rbeta, surface=surface,
        colormap=mpl.colormaps.get_cmap('viridis'), k=0.5
        )