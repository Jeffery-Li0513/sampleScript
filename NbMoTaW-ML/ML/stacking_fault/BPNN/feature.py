import pandas as pd
from matminer.featurizers.composition import alloy, thermo
from matminer.featurizers.structure import bonding
from matminer.featurizers.conversions import StrToComposition
from sklearn.preprocessing import StandardScaler
import torch.utils.data as Data
import torch

MP_API_KEY = "3NTu15Jeug12iKJAyM8EgliBL6BgKO3K"


def featurize(data_path):
    data = pd.read_csv(data_path)
    # Based on the composition of the alloy
    # data = StrToComposition(target_col_id="composition").featurize_dataframe(data, "formula")
    # data = alloy.WenAlloys().featurize_dataframe(data, "composition")
    # Based on the structure of the alloy

    data.dropna(axis=1, how='any', inplace=True)
    data.loc[:, (data != data.iloc[0]).any()]
    data.drop(['formula', 'composition', 'Weight Fraction', 'Atomic Fraction', 'Interant electrons', 
                        'Interant s electrons', 'Interant p electrons', 'Interant d electrons', 'Interant f electrons'], axis=1, inplace=True)
    # 随机森林特征重要性筛选得到
    data_fit = data[['VEC mean', 'APE mean', 'Electronegativity local mismatch',
       'Shear modulus delta', 'Shear modulus mean', 'Mixing enthalpy',
       'Shear modulus strength model', 'Nb', 'Mo', 'Radii gamma',
       'Yang omega', 'Electronegativity delta', 'Lambda entropy', 'Ta',
       'Shear modulus local mismatch', 'Configuration entropy',
       'Yang delta', 'W', 'Radii local mismatch', 'Mean cohesive energy',
       'Total weight', 'Atomic weight mean']]

    # data.to_csv('data_featurized.csv', index=False)
    data_fit.to_csv('dataset.csv', index=False)

def convet_to_tenserdataset(data):
    # data = pd.read_csv(data)
    X = data.drop(['SFE'], axis=1)
    Y = data['SFE']
    tensor_dataset = Data.TensorDataset(torch.FloatTensor(X.values), torch.FloatTensor(Y.values))
    dataset = Data.DataLoader(tensor_dataset)
    return dataset

def dataset_split(dataset, train=0.6, val=0.2, test=0.2):
    data = pd.read_csv(dataset)
    train_data = data.iloc[:int(len(data)*train)]
    val_data = data.iloc[int(len(data)*train):int(len(data)*(train+val))]
    test_data = data.iloc[int(len(data)*(train+val)):]
    print(len(train_data), len(val_data), len(test_data))
    train_data.to_csv('train.csv', index=False)
    val_data.to_csv('val.csv', index=False)
    test_data.to_csv('test.csv', index=False)
    # return train_data, val_data, test_data

if __name__ == '__main__':
    # featurize(data_path='data_featurized.csv')
    dataset_split(dataset='dataset.csv')