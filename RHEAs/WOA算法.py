
import numpy as np
import random
from KMC import neighborlist
import KMC

dim = 80
b = 1
whale_num = 6
max_iter = 500

# initialize the locations of whales
X = []
for whale in range(whale_num):
    atom_types = []
    for i in range(dim):
        atom_types.append(random.randint(1, 5))
    X.append(atom_types)
X = np.array(X)

gBest_coeff = 1
gBest_X = np.zeros(dim)
gBest_curve = np.zeros(max_iter)
nei_list = neighborlist('equiHEA1')
gBest_solu_ener = []
t = 0
random.seed(19)
while t < max_iter:
    # update best whale and best coefficient
    for i in range(whale_num):
        for ele in range(dim):
            if X[i, ele] > 5 or X[i, ele] < 1:
                X[i, ele] = random.randint(1, 5)
        x = [0, 0, 0, 0, 0]
        for atom_type in X[i, :]:
            x[int(atom_type) - 1] += 1
        if x[0] * x[1] * x[2] * x[3] * x[4] == 0:
            atom_types = []
            for j in range(dim):
                atom_types.append(random.randint(1, 5))
            X[i, :] = np.array(atom_types)
        solu_ener = solution_energy(X[i, :])  # use the trained ML model to predict H solution energies

        fitness = kMC(nei_list, solu_ener)
        if round(np.log(fitness), 2) <= round(np.log(gBest_coeff), 2):  # to avoid falling into local minimum
            gBest_coeff = fitness
            gBest_X = X[i, :].copy()
            gBest_solu_ener = solu_ener

    a = 2 * (max_iter - t) / max_iter
    # update the whales
    for i in range(whale_num):
        p = np.random.uniform()
        R1 = np.random.uniform()
        R2 = np.random.uniform()
        A = 2 * a * R1 - a
        C = 2 * R2
        l = 2 * np.random.uniform() - 1

        if p >= 0.5:
            D = abs(gBest_X - X[i, :])
            X[i, :] = D * np.exp(b * l) * np.cos(2 * np.pi * l) + gBest_X
        else:
            if abs(A) < 1:
                D = abs(C * gBest_X - X[i, :])
                X[i, :] = gBest_X - A * D
            else:
                rand_index = np.random.randint(low=0, high=whale_num)
                X_rand = X[rand_index, :]
                D = abs(C * X_rand - X[i, :])
                X[i, :] = X_rand - A * D



x_arr = np.load('compositions.npy') #the atomic ratios of each element in the HEA
y_arr = np.load('diffusion_coefficients.npy')  # the corresponding H diffusion coefficients
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_arr, y_arr, test_size=0.2, random_state=42)
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

nn = RidgeCV()

nn.fit(x_train, y_train)
w = nn.coef_

import matplotlib.pyplot as plt

low = min(y_arr) - 0.1
high = max(y_arr) + 0.1
lims = [-16.9, -13]

text = 'Linear regression'
plt.figure(2, figsize=(8, 8))
plt.plot(lims, lims, '--', linewidth=2, color='black')
coeff_predict = nn.predict(x_train)
plt.plot(y_train, coeff_predict, '+', color='orangered', alpha=0.5)
plt.xlim(lims)
plt.ylim(lims)

plt.text(-16.8, -13.2, text, fontsize=22)
plt.text(-16.8, -13.5, 'Training set', fontsize=22)
plt.text(-16.8, -13.8, '$\mathregular{R^2}$ =' + str(round(r2_score(y_train, coeff_predict), 2)), fontsize=22)
plt.ylabel('Predicted value', fontsize=20)
plt.xlabel('True value', fontsize=20)
a = [-16.5, -16, -15.5, -15, -14.5, -14, -13.5, -13]
plt.xticks(a, fontsize=18)
plt.yticks(a, fontsize=18)

plt.figure(3, figsize=(8, 8))
plt.plot(lims, lims, '--', linewidth=2, color='black')
coeff_predict = nn.predict(x_test)
plt.plot(y_test, coeff_predict, '+', color='deeppink', alpha=0.5)
plt.xlim(lims)
plt.ylim(lims)

plt.text(-16.8, -13.2, text, fontsize=22)
plt.text(-16.8, -13.5, 'Test set', fontsize=22)
plt.text(-16.8, -13.8, '$\mathregular{R^2}$ =' + str(round(r2_score(y_test, coeff_predict), 2)), fontsize=22)
plt.ylabel('Predicted value', fontsize=20)
plt.xlabel('True value', fontsize=20)
a = [-16.5, -16, -15.5, -15, -14.5, -14, -13.5, -13]
plt.xticks(a, fontsize=18)
plt.yticks(a, fontsize=18)

# 2-degree polynomial regression
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(interaction_only=True, degree=2)
poly.fit(x_arr)
x_arr = poly.transform(x_arr)

x_train, x_test, y_train, y_test = train_test_split(x_arr, y_arr, test_size=0.2, random_state=42)
nn = RidgeCV()
nn.fit(x_train, y_train)
w = nn.coef_

import matplotlib.pyplot as plt

low = min(y_arr) - 0.1
high = max(y_arr) + 0.1
lims = [-16.9, -13]

text = 'Degree-2 polynomial regression'
plt.figure(2, figsize=(8, 8))
plt.plot(lims, lims, '--', linewidth=2, color='black')
coeff_predict = nn.predict(x_train)
plt.plot(y_train, coeff_predict, '+', color='royalblue', alpha=0.5)
plt.xlim(lims)
plt.ylim(lims)

plt.text(-16.8, -13.2, text, fontsize=22)
plt.text(-16.8, -13.5, 'Training set', fontsize=22)
plt.text(-16.8, -13.8, '$\mathregular{R^2}$ =' + str(round(r2_score(y_train, coeff_predict), 2)), fontsize=22)
plt.ylabel('Predicted value', fontsize=20)
plt.xlabel('True value', fontsize=20)
a = [-16.5, -16, -15.5, -15, -14.5, -14, -13.5, -13]
plt.xticks(a, fontsize=18)
plt.yticks(a, fontsize=18)

plt.figure(3, figsize=(8, 8))
plt.plot(lims, lims, '--', linewidth=2, color='black')
coeff_predict = nn.predict(x_test)
plt.plot(y_test, coeff_predict, '+', color='darkviolet', alpha=0.5)
plt.xlim(lims)
plt.ylim(lims)

plt.text(-16.8, -13.2, text, fontsize=22)
plt.text(-16.8, -13.5, 'Test set', fontsize=22)
plt.text(-16.8, -13.8, '$\mathregular{R^2}$=' + str(round(r2_score(y_test, coeff_predict), 2)), fontsize=22)
plt.ylabel('Predicted value', fontsize=20)
plt.xlabel('True value', fontsize=20)
a = [-16.5, -16, -15.5, -15, -14.5, -14, -13.5, -13]
plt.xticks(a, fontsize=18)
plt.yticks(a, fontsize=18)
