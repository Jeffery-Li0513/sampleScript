import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from torchmetrics import R2Score
from scipy import stats


class BPNN(nn.Module):
    def __init__(self, n_feature, hidden_neuron_list, n_output, n_layers, dropout=False, dropout_rate=0.5, active_function_type='Relu'):
        super(BPNN, self).__init__()
        self.n_feature = n_feature
        self.hidden_neuron_list = hidden_neuron_list
        self.active_function_type = active_function_type
        self.n_output = n_output
        self.n_layers = n_layers
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.input = nn.Linear(self.n_feature, self.hidden_neuron_list[0])
        self.modulelist = nn.ModuleList()
        for i in range(len(hidden_neuron_list)-1):
            if self.dropout:
                self.modulelist.append(nn.Dropout(self.dropout_rate))
            self.modulelist.append(nn.Linear(self.hidden_neuron_list[i], self.hidden_neuron_list[i+1]))
            self.modulelist.append(nn.ReLU())
        self.predict = nn.Linear(self.hidden_neuron_list[-1], self.n_output)

    def forward(self, x):
        out = self.input(x)
        for module in self.modulelist:
            out = module(out)
        out = self.predict(out)

        return out
    
    def active_function(self, out, active_function_type='ReLU'):
        if active_function_type == 'ReLU':
            return nn.functional.relu(out)
        elif active_function_type == 'Sigmoid':
            return nn.functional.sigmoid(out)
        elif active_function_type == 'Tanh':
            return nn.functional.tanh(out)
        elif active_function_type == 'Softmax':
            return nn.functional.softmax(out)
        elif active_function_type == 'LeakyReLU':
            return nn.functional.leaky_relu(out)
        elif active_function_type == 'Softplus':
            return nn.functional.softplus(out)

# Path: ML\stacking_fault\BPNN\train.py

def regularization_(loss, model, lamda, regularization_type):
    if regularization_type == 'L1':
        loss = loss + lamda * sum(p.abs().sum() for p in model.parameters())
    elif regularization_type == 'L2':
        loss = loss + lamda * sum(p.pow(2.0).sum() for p in model.parameters())
    elif regularization_type == 'Dropout':
        return loss
    return loss


def train(model, train_loader, validation_loader, optimizer, criterion, epochs, device, regularization=False, regularization_type=None, lamda=None):
    train_loss = []
    val_loss = []
    val_r2 = []
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            print(data.device)
            y_train_pred = model.forward(data)
            loss = criterion(y_train_pred.squeeze(1), target)
            if regularization:
                loss = regularization_(loss, model, lamda, regularization_type)
            optimizer.zero_grad()               # 梯度清零
            loss.backward()
            optimizer.step()
        train_loss.append(loss.item())
        # print("train epoch %d, loss %s:" % (epoch + 1, loss.item()))
        model.eval()
        y_val_pred = []
        y_val_true = []
        for data, target in validation_loader:                                                  # validation_loader不需要添加batch_size
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            y_pred = model.forward(data).detach().cpu().numpy()
            target = target.detach().cpu().numpy()
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
    if device == "cuda":
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


def lineregress(actual, pred):
    slope, intercept, r_value, p_value, std_err = stats.linregress(actual, pred)        # 斜率，截距，Pearson相关系数，p值，标准误差
    rmse = np.sqrt(np.mean((actual - pred) ** 2))
    mae = np.mean(np.abs(actual - pred))

    return r_value**2, rmse, 1-rmse, mae, 1-mae


def test_best_feature_combinations():
    feature_combinations = feature_combination(5, train_data)
    for features in feature_combinations[0][:1]:
        # features = list(features)
        features = ['APE mean', 'Electronegativity local mismatch', 'VEC mean', 'Shear modulus mean', 'Shear modulus delta', 'Shear modulus strength model']
        features = ['VEC mean', 'APE mean', 'Electronegativity local mismatch', 'Shear modulus delta', 'Mo']
        train_loader = convert_to_dataloader(train_data, features, batch_size=64)
        val_loader = convert_to_dataloader(val_data, features)
        test_loader = convert_to_dataloader(test_data, features)

        bpnn = BPNN(n_feature=len(features), n_hidden1=20, n_hidden2=20, n_output=1, n_layers=2)
        optimizer = torch.optim.Adam(bpnn.parameters(), lr=0.001)
        criterion=nn.MSELoss()
        train_loss, val_loss = train(model=bpnn, train_loader=train_loader, validation_loader=val_loader, optimizer=optimizer, criterion=criterion, epochs=1000)
        results.loc[len(results)] = [features, train_loss, val_loss]


def test_best_features_num(train_data, val_data, device):
    '''
    找出最佳的特征数目，需要传入已经重要性排序过的特征列表；以及训练集、验证集
    '''
    feature_list = ['VEC mean', 'APE mean', 'Electronegativity local mismatch',
       'Shear modulus delta', 'Shear modulus mean', 'Mixing enthalpy',
       'Shear modulus strength model', 'Nb', 'Mo', 'Radii gamma',
       'Yang omega', 'Electronegativity delta', 'Lambda entropy', 'Ta',
       'Shear modulus local mismatch', 'Configuration entropy',
       'Yang delta', 'W', 'Radii local mismatch', 'Mean cohesive energy',
       'Total weight', 'Atomic weight mean']
    for i in range(3, 11):
        results = pd.DataFrame(columns=['features', 'train_loss', 'val_loss', 'val_r2'])

        features = feature_list[:i]
        train_loader = convert_to_dataloader(train_data, features, batch_size=64)
        val_loader = convert_to_dataloader(val_data, features)

        bpnn = BPNN(n_feature=len(features), hidden_neuron_list=[20, 20], n_output=1, n_layers=2)
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


def test_best_neuron_layers(hidden_layers, features, train_data, val_data, device):
    # 测试不同层数的神经网络
    for n_layers in hidden_layers:
        results = pd.DataFrame(columns=['n_layers', 'train_loss', 'val_loss', 'val_r2'])
        
        train_loader = convert_to_dataloader(train_data, features, batch_size=64, device=device)
        val_loader = convert_to_dataloader(val_data, features, device=device)

        hidden_neuron_list = [20]*n_layers
        bpnn = BPNN(n_feature=len(features), hidden_neuron_list=hidden_neuron_list, n_output=1, n_layers=n_layers)
        bpnn.to(device)
        optimizer = torch.optim.Adam(bpnn.parameters(), lr=0.001)
        criterion=nn.MSELoss()
        train_loss, val_loss, val_r2 = train(model=bpnn, train_loader=train_loader, validation_loader=val_loader, optimizer=optimizer, criterion=criterion, epochs=1000)
        results.loc[len(results)] = [n_layers, train_loss, val_loss, val_r2]

        results.to_csv('./results/{}_layers.csv'.format(n_layers))
        # 输出最佳结果
        min_train_loss = min(results.loc[0, 'train_loss'])
        min_val_loss = min(results.loc[0, 'val_loss'])
        max_val_r2 = max(results.loc[0, 'val_r2'])
        print(results.loc[0, 'n_layers'], min_train_loss, min_val_loss, max_val_r2)


def test_best_neuron_number():
    features = ['VEC mean', 'APE mean', 'Electronegativity local mismatch',
       'Shear modulus delta', 'Shear modulus mean', 'Mixing enthalpy',
       'Shear modulus strength model']
    n_layers = 3
    hidden_neuron_list=[20, 20 , 20]
    # 第一个隐藏层神经元个数
    for i in range(1, 11):
        results = pd.DataFrame(columns=['1_layer_neuronnum', 'train_loss', 'val_loss', 'val_r2'])
        train_loader = convert_to_dataloader(train_data, features, batch_size=64, device=device)
        val_loader = convert_to_dataloader(val_data, features, device=device)

        hidden_neuron_list[0] = i
        bpnn = BPNN(n_feature=len(features), hidden_neuron_list=hidden_neuron_list, n_output=1, n_layers=n_layers)
        bpnn.to(device)
        optimizer = torch.optim.Adam(bpnn.parameters(), lr=0.001)
        criterion=nn.MSELoss()
        train_loss, val_loss, val_r2 = train(model=bpnn, train_loader=train_loader, validation_loader=val_loader, optimizer=optimizer, criterion=criterion, epochs=1000)
        results.loc[len(results)] = [i, train_loss, val_loss, val_r2]

        results.to_csv('./results/1_layer_{}_neuron.csv'.format(i))
        # 输出最佳结果
        min_train_loss = min(results.loc[0, 'train_loss'])
        min_val_loss = min(results.loc[0, 'val_loss'])
        max_val_r2 = max(results.loc[0, 'val_r2'])
        print(results.loc[0, '1_layer_neuronnum'], min_train_loss, min_val_loss, max_val_r2)


def test_regularization():
    for i in ['None', 'L1', 'L2', 'Dropout']:
        results = pd.DataFrame(columns=['regularization', 'train_loss', 'val_loss', 'val_r2'])
        train_loader = convert_to_dataloader(train_data, features, batch_size=64, device=device)
        val_loader = convert_to_dataloader(val_data, features, device=device)

        if i == 'Dropout':
            bpnn = BPNN(n_feature=len(features), hidden_neuron_list=hidden_neuron_list, n_output=1, n_layers=n_layers, dropout=True, dropout_rate=0.1)
        else:
            bpnn = BPNN(n_feature=len(features), hidden_neuron_list=hidden_neuron_list, n_output=1, n_layers=n_layers)
        bpnn.to(device)
        optimizer = torch.optim.Adam(bpnn.parameters(), lr=0.001)
        criterion=nn.MSELoss()
        if i == 'None':
            train_loss, val_loss, val_r2 = train(model=bpnn, train_loader=train_loader, validation_loader=val_loader, optimizer=optimizer, 
                                                 criterion=criterion, epochs=1000, device=device)
        else:
            train_loss, val_loss, val_r2 = train(model=bpnn, train_loader=train_loader, validation_loader=val_loader, optimizer=optimizer, 
                                             criterion=criterion, epochs=1000, device=device, regularization=True, regularization_type=i, lamda=0.001)
        results.loc[len(results)] = [i, train_loss, val_loss, val_r2]

        results.to_csv('../results/1500/{}_regularization.csv'.format(i))
        # 输出最佳结果
        min_train_loss = min(results.loc[0, 'train_loss'])
        min_val_loss = min(results.loc[0, 'val_loss'])
        max_val_r2 = max(results.loc[0, 'val_r2'])
        print(results.loc[0, 'regularization'], min_train_loss, min_val_loss, max_val_r2)


def test_best_active_function():
    active_function = ['ReLU', 'LeakyReLU', 'Tanh', 'Sigmoid', 'Softmax', 'Softplus']
    for i in active_function:
        results = pd.DataFrame(columns=['active_function', 'train_loss', 'val_loss', 'val_r2'])
        train_loader = convert_to_dataloader(train_data, features, batch_size=64, device=device)
        val_loader = convert_to_dataloader(val_data, features, device=device)

        bpnn = BPNN(n_feature=len(features), hidden_neuron_list=hidden_neuron_list, n_output=1, n_layers=n_layers, active_function_type=i)
        bpnn.to(device)
        optimizer = torch.optim.Adam(bpnn.parameters(), lr=0.001)
        criterion=nn.MSELoss()
        train_loss, val_loss, val_r2 = train(model=bpnn, train_loader=train_loader, validation_loader=val_loader, optimizer=optimizer, 
                                             criterion=criterion, epochs=1000, device=device, regularization=True, regularization_type=regularization_type, 
                                             lamda=0.001)
        results.loc[len(results)] = [i, train_loss, val_loss, val_r2]
        results.to_csv('../results/1500/{}_active.csv'.format(i))
        # 输出最佳结果
        min_train_loss = min(results.loc[0, 'train_loss'])
        min_val_loss = min(results.loc[0, 'val_loss'])
        max_val_r2 = max(results.loc[0, 'val_r2'])
        print(results.loc[0, 'active_function'], min_train_loss, min_val_loss, max_val_r2)


if __name__ == '__main__':
    train_data = pd.read_csv('../data/1500/train.csv')
    val_data = pd.read_csv('../data/1500/val.csv')
    test_data = pd.read_csv('../data/1500/test.csv')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    features = ['VEC mean', 'APE mean', 'Electronegativity local mismatch',
    'Shear modulus delta', 'Shear modulus mean', 'Mixing enthalpy',
    'Shear modulus strength model']
    n_layers = 3
    hidden_neuron_list=[18, 18, 18]
    regularization_type = 'L2'
    active_function = 'ReLU'

    # 最优特征数目
    # test_best_features_num(train_data, val_data, device)

    # 最优特征组合
    # test_best_feature_combinations()

    # 优化神经网络的层数
    # hidden_layers = [1, 2, 3, 4]
    # test_best_neuron_layers(hidden_layers, features, train_data, val_data, device)

    # 优化三隐藏层神经网络的超参数,batch_size, learning_rate, active function, 隐藏层神经元个数
    # test_best_neuron_number()
    # 正则化: L1 L2 Dropout
    # test_regularization()
    # 优化激活函数
    # test_best_active_function()
    