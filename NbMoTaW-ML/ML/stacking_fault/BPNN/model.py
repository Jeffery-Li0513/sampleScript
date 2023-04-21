import torch
import torch.nn as nn
import pandas as pd
from itertools import combinations
from sklearn.preprocessing import StandardScaler
import numpy as np
from torchmetrics import R2Score

class BPNN(nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_output, n_layers):
        super(BPNN, self).__init__()
        self.n_feature = n_feature
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_output = n_output
        self.n_layers = n_layers
        self.input_layer = nn.Linear(self.n_feature, self.n_hidden1)
        self.hidden1 = nn.Linear(self.n_hidden1, self.n_hidden2)
        self.hidden2 = nn.Linear(self.n_hidden2, self.n_hidden2)
        self.predict = nn.Linear(n_hidden2, n_output)

    def forward(self, x):
        out = self.input_layer(x)
        out = nn.functional.relu(out)
        out = self.hidden1(out)
        out = nn.functional.relu(out)
        if self.n_layers == 2:
            out = self.hidden2(out)
            out = nn.functional.relu(out)
        out = self.predict(out)

        return out

# Path: ML\stacking_fault\BPNN\train.py

def train(model, train_loader, validation_loader, optimizer, criterion, epochs):
    train_loss = []
    val_loss = []
    val_r2 = []
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            y_train_pred = model.forward(data)
            loss = criterion(y_train_pred.squeeze(1), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss.append(loss.item())
        # print("train epoch %d, loss %s:" % (epoch + 1, loss.item()))
        model.eval()
        y_val_pred = []
        y_val_true = []
        for data, target in validation_loader:                                                  # validation_loader不需要添加batch_size
            y_pred = model.forward(data).detach().numpy()
            target = target.detach().numpy()
            y_val_pred.append(y_pred)
            y_val_true.append(target)
        average_loss = criterion(torch.FloatTensor(np.array(y_val_pred)).squeeze(1), torch.FloatTensor(np.array(y_val_true)))
        r2 = R2Score()(torch.FloatTensor(np.array(y_val_pred)).squeeze(1), torch.FloatTensor(np.array(y_val_true)))
        val_loss.append(average_loss.item())
        val_r2.append(r2.item())
        # print("validation epoch %d, loss %s:" % (epoch + 1, loss.item()))
        # print(R2Score()(torch.FloatTensor(np.array(y_val_pred)).squeeze(1), torch.FloatTensor(np.array(y_val_true))))
    
    return train_loss, val_loss, val_r2


def feature_combination(feature_numbers, dataset):
    feature_list = dataset.columns.tolist()
    feature_list.pop(feature_list.index('SFE'))
    feature_combinations = [list(combinations(feature_list, feature_numbers))]
    # print(len(feature_combinations[0]))

    return feature_combinations


def convert_to_dataloader(dataset, features, batch_size=None, device=None):
    X = dataset[features]
    Y = dataset['SFE']
    standard_scaler = StandardScaler()
    X = standard_scaler.fit_transform(X)
    if device is not None:
        X = torch.FloatTensor(X).cuda(device=device)
        Y = torch.FloatTensor(Y).cuda(device=device)
    else:
        X = torch.FloatTensor(X)
        Y = torch.FloatTensor(Y)
    torch_dataset = torch.utils.data.TensorDataset(X, Y)
    if batch_size is not None:
        dataset = torch.utils.data.DataLoader(torch_dataset, batch_size=batch_size, shuffle=False)
    else:
        dataset = torch.utils.data.DataLoader(torch_dataset, shuffle=False)

    return dataset


def test_best_features_num(feature_list, train_data, val_data, device):
    '''
    找出最佳的特征数目，需要传入已经重要性排序过的特征列表；以及训练集、验证集
    '''
    for i in range(3, 11):
        results = pd.DataFrame(columns=['features', 'train_loss', 'val_loss', 'val_r2'])

        features = feature_list[:i]
        train_loader = convert_to_dataloader(train_data, features, batch_size=64)
        val_loader = convert_to_dataloader(val_data, features)

        bpnn = BPNN(n_feature=len(features), n_hidden1=20, n_hidden2=20, n_output=1, n_layers=2)
        bpnn.to(device)
        optimizer = torch.optim.Adam(bpnn.parameters(), lr=0.001)
        criterion=nn.MSELoss()
        train_loss, val_loss, val_r2 = train(model=bpnn, train_loader=train_loader, validation_loader=val_loader, optimizer=optimizer, criterion=criterion, epochs=1000)
        results.loc[len(results)] = [features, train_loss, val_loss, val_r2]

        results.to_csv('./results/{}.csv'.format('_'.join([i.split()[0] for i in features])))
        # 输出最佳结果
        min_train_loss = min(results.loc[0, 'train_loss'])
        min_val_loss = min(results.loc[0, 'val_loss'])
        max_val_r2 = max(results.loc[0, 'val_r2'])
        print(results.loc[0, 'features'], min_train_loss, min_val_loss, max_val_r2)


if __name__ == '__main__':
    train_data = pd.read_csv('data/train.csv')
    val_data = pd.read_csv('data/val.csv')
    test_data = pd.read_csv('data/test.csv')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_list = ['VEC mean', 'APE mean', 'Electronegativity local mismatch',
       'Shear modulus delta', 'Shear modulus mean', 'Mixing enthalpy',
       'Shear modulus strength model', 'Nb', 'Mo', 'Radii gamma',
       'Yang omega', 'Electronegativity delta', 'Lambda entropy', 'Ta',
       'Shear modulus local mismatch', 'Configuration entropy',
       'Yang delta', 'W', 'Radii local mismatch', 'Mean cohesive energy',
       'Total weight', 'Atomic weight mean']
    
    test_best_features_num(feature_list, train_data, val_data, device)

    # feature_combinations = feature_combination(5, train_data)
    # for features in feature_combinations[0][:1]:
    #     # features = list(features)
    #     features = ['APE mean', 'Electronegativity local mismatch', 'VEC mean', 'Shear modulus mean', 'Shear modulus delta', 'Shear modulus strength model']
    #     features = ['VEC mean', 'APE mean', 'Electronegativity local mismatch', 'Shear modulus delta', 'Mo']
    #     train_loader = convert_to_dataloader(train_data, features, batch_size=64)
    #     val_loader = convert_to_dataloader(val_data, features)
    #     test_loader = convert_to_dataloader(test_data, features)

    #     bpnn = BPNN(n_feature=len(features), n_hidden1=20, n_hidden2=20, n_output=1, n_layers=2)
    #     optimizer = torch.optim.Adam(bpnn.parameters(), lr=0.001)
    #     criterion=nn.MSELoss()
    #     train_loss, val_loss = train(model=bpnn, train_loader=train_loader, validation_loader=val_loader, optimizer=optimizer, criterion=criterion, epochs=1000)
    #     results.loc[len(results)] = [features, train_loss, val_loss]

    # 优化神经网络的超参数
    