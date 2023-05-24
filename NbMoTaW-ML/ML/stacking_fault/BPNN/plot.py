import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

title_font = {"fontsize": 24,}
label_font = {"fontsize": 22,}
text_font = {"fontsize": 18,}
pad = 20
epochs = 1000
figure_size = (10, 7)
dpi = 800
ticks_size = 20
legend_size = 10


# 画出训练集和测试集的MESLOSS变化曲线
def plot_loss(i):
    fig_train_loss = plt.figure(figsize=figure_size, dpi=dpi)

    xlim = (-10, 1000)
    ylim = (0, 0.01)
    xticks = (True, (0, 200, 400, 600, 800, 1000))
    yticks = (True, tuple([i for i in np.arange(0, 0.011, 0.002)]))
    num1, num2, num3, num4 = 0.99, 0.99, 1, 0          # 用于调整图例的位置，使其不遮挡图像。num1和num2分别是图例的x和y坐标，num3控制图例位置。
    tx = np.arange(0, epochs, 1)
    ylabel_loss = 'MSE' + r'$\/(\frac{J}{m^2})$'
    xlabel = 'Epoch'
    title_train_loss = 'Training MSEloss vs Epoch on different regularization'

    plt.title(title_train_loss, fontdict=title_font, weight='bold', pad=pad)

    result = pd.read_csv('./results/1500/{}_regularization.csv'.format(i))
    train_loss = np.array(eval(result.loc[0, 'train_loss']))
    val_loss = np.array(eval(result.loc[0, 'val_loss']))
    plt.plot(tx, train_loss, label='{}'.format(i))
    plt.plot(tx, val_loss, label='{}'.format(i))
    
    plt.legend(bbox_to_anchor=(num1, num2), title=None, loc=num3, fontsize=legend_size, ncol=1)
    plt.xlabel(xlabel, fontdict=label_font, weight='bold')
    plt.ylabel(ylabel_loss, fontdict=label_font, weight='bold')
    plt.xticks(xticks[1], fontsize=ticks_size, weight='bold')
    plt.yticks(yticks[1], fontsize=ticks_size, weight='bold')
    plt.tick_params(labelsize=ticks_size, length=5, width=1.4)
    tk = plt.gca()
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.subplots_adjust(top=0.89,
                    bottom=0.16,
                    left=0.15,
                    right=0.85,
                    hspace=0.2,
                    wspace=0.2)

    fig_train_loss.savefig("./loss.png")


# 画出训练集和测试集的R2变化曲线


# 随特征数的增加模型的LOSS变化曲线
def plot_train_loss_with_feature_number(feature_list):
    fig_train_loss = plt.figure(figsize=figure_size, dpi=dpi)

    xlim = (-10, 1000)
    ylim = (0, 0.01)
    xticks = (True, (0, 200, 400, 600, 800, 1000))
    yticks = (True, tuple([i for i in np.arange(0, 0.011, 0.002)]))
    num1, num2, num3, num4 = 0.99, 0.99, 1, 0          # 用于调整图例的位置，使其不遮挡图像。num1和num2分别是图例的x和y坐标，num3控制图例位置。
    tx = np.arange(0, epochs, 1)
    ylabel_loss = 'MSE' + r'$\/(\frac{J}{m^2})$'
    xlabel = 'Epoch'
    title_train_loss = 'Training MSEloss vs Epoch on different features'

    plt.title(title_train_loss, fontdict=title_font, weight='bold', pad=pad)

    for i in range(3, 11):
        features = feature_list[:i]
        result = pd.read_csv('./results/{}.csv'.format('_'.join([i.split()[0] for i in features])))
        train_loss = np.array(eval(result.loc[0, 'train_loss']))
        plt.plot(tx, train_loss, label='{}'.format('_'.join([i.split()[0] for i in features])))
    
    plt.legend(bbox_to_anchor=(num1, num2), title=None, loc=num3, fontsize=legend_size, ncol=1)
    plt.xlabel(xlabel, fontdict=label_font, weight='bold')
    plt.ylabel(ylabel_loss, fontdict=label_font, weight='bold')
    plt.xticks(xticks[1], fontsize=ticks_size, weight='bold')
    plt.yticks(yticks[1], fontsize=ticks_size, weight='bold')
    plt.tick_params(labelsize=ticks_size, length=5, width=1.4)
    tk = plt.gca()
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.subplots_adjust(top=0.89,
                    bottom=0.16,
                    left=0.15,
                    right=0.85,
                    hspace=0.2,
                    wspace=0.2)

    fig_train_loss.savefig("./images/train_loss.png")


def plot_val_loss_with_feature_number(feature_list):
    fig_val_loss = plt.figure(figsize=figure_size, dpi=dpi)

    xlim = (-10, 1000)
    ylim = (0, 0.1)
    xticks = (True, (0, 200, 400, 600, 800, 1000))
    yticks = (True, tuple([i for i in np.arange(0, 0.11, 0.02)]))
    num1, num2, num3, num4 = 0.99, 0.99, 1, 0          # 用于调整图例的位置，使其不遮挡图像。num1和num2分别是图例的x和y坐标，num3控制图例位置。
    tx = np.arange(0, epochs, 1)
    ylabel_loss = 'MSE' + r'$\/(\frac{J}{m^2})$'
    xlabel = 'Epoch'
    title_val_loss = 'Validation MSEloss vs Epoch on different features'

    plt.title(title_val_loss, fontdict=title_font, weight='bold', pad=pad)

    for i in range(3, 11):
        features = feature_list[:i]
        result = pd.read_csv('./results/{}.csv'.format('_'.join([i.split()[0] for i in features])))
        val_loss = np.array(eval(result.loc[0, 'val_loss']))
        plt.plot(tx, val_loss, label='{}'.format('_'.join([i.split()[0] for i in features])))

    plt.legend(bbox_to_anchor=(num1, num2), loc=num3, title=None, fontsize=legend_size)
    plt.xlabel(xlabel, fontdict=label_font, weight='bold')
    plt.ylabel(ylabel_loss, fontdict=label_font, weight='bold')
    plt.xticks(xticks[1], fontsize=ticks_size, weight='bold')
    plt.yticks(yticks[1], fontsize=ticks_size, weight='bold')
    plt.tick_params(labelsize=ticks_size, length=5, width=1.4)
    tk = plt.gca()
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.subplots_adjust(top=0.89,
                    bottom=0.16,
                    left=0.15,
                    right=0.85,
                    hspace=0.2,
                    wspace=0.2)

    fig_val_loss.savefig("./images/val_loss.png")
    

def plot_val_r2_with_feature_number(feature_list):
    fig_val_r2 = plt.figure(figsize=figure_size, dpi=dpi)

    xlim = (-10, 1000)
    ylim = (0.5, 1)
    xticks = (True, (0, 200, 400, 600, 800, 1000))
    yticks = (True, tuple([i for i in np.arange(0.5, 1.01, 0.1)]))
    num1, num2, num3, num4 = 0.99, 0.5, 1, 0
    tx = np.arange(0, epochs, 1)
    ylabel_r2 = r'$R^2$'
    xlabel = 'Epoch'
    title_val_r2 = 'Validation R2 vs Epoch on different features'

    plt.title(title_val_r2, fontdict=title_font, weight='bold', pad=pad)

    for i in range(3, 11):
        features = feature_list[:i]
        result = pd.read_csv('./results/{}.csv'.format('_'.join([i.split()[0] for i in features])))
        val_r2 = np.array(eval(result.loc[0, 'val_r2']))
        plt.plot(tx, val_r2, label='{}'.format('_'.join([i.split()[0] for i in features])))

    plt.legend(bbox_to_anchor=(num1, num2), loc=num3, title=None, fontsize=legend_size)
    plt.xlabel(xlabel, fontdict=label_font, weight='bold')
    plt.ylabel(ylabel_r2, fontdict=label_font, weight='bold')
    plt.xticks(xticks[1], fontsize=ticks_size, weight='bold')
    plt.yticks(yticks[1], fontsize=ticks_size, weight='bold')
    plt.tick_params(labelsize=ticks_size, length=5, width=1.4)
    tk = plt.gca()
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.subplots_adjust(top=0.89,
                    bottom=0.16,
                    left=0.15,
                    right=0.85,
                    hspace=0.2,
                    wspace=0.2)

    fig_val_r2.savefig("./images/val_r2.png")


def plot_train_loss_neuron_number(layers_list):
    fig_train_loss = plt.figure(figsize=figure_size, dpi=dpi)

    xlim = (-10, 1000)
    ylim = (0, 0.01)
    xticks = (True, (0, 200, 400, 600, 800, 1000))
    yticks = (True, tuple([i for i in np.arange(0, 0.011, 0.002)]))
    num1, num2, num3, num4 = 0.99, 0.99, 1, 0          # 用于调整图例的位置，使其不遮挡图像。num1和num2分别是图例的x和y坐标，num3控制图例位置。
    tx = np.arange(0, epochs, 1)
    ylabel_loss = 'MSE' + r'$\/(\frac{J}{m^2})$'
    xlabel = 'Epoch'
    title_train_loss = 'Training MSEloss vs Epoch on different layers'

    plt.title(title_train_loss, fontdict=title_font, weight='bold', pad=pad)

    for i in layers_list:
        result = pd.read_csv('./results/{}_layers.csv'.format(i))
        train_loss = np.array(eval(result.loc[0, 'train_loss']))
        plt.plot(tx, train_loss, label='{}_layers'.format(i))
    
    plt.legend(bbox_to_anchor=(num1, num2), title=None, loc=num3, fontsize=legend_size, ncol=1)
    plt.xlabel(xlabel, fontdict=label_font, weight='bold')
    plt.ylabel(ylabel_loss, fontdict=label_font, weight='bold')
    plt.xticks(xticks[1], fontsize=ticks_size, weight='bold')
    plt.yticks(yticks[1], fontsize=ticks_size, weight='bold')
    plt.tick_params(labelsize=ticks_size, length=5, width=1.4)
    tk = plt.gca()
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.subplots_adjust(top=0.89,
                    bottom=0.16,
                    left=0.15,
                    right=0.85,
                    hspace=0.2,
                    wspace=0.2)

    fig_train_loss.savefig("./images/train_loss_on_layers.png")


def plot_val_loss_neuron_number(layers_list):
    fig_train_loss = plt.figure(figsize=figure_size, dpi=dpi)

    xlim = (-10, 1000)
    ylim = (0, 0.1)
    xticks = (True, (0, 200, 400, 600, 800, 1000))
    yticks = (True, tuple([i for i in np.arange(0, 0.11, 0.02)]))
    num1, num2, num3, num4 = 0.99, 0.99, 1, 0          # 用于调整图例的位置，使其不遮挡图像。num1和num2分别是图例的x和y坐标，num3控制图例位置。
    tx = np.arange(0, epochs, 1)
    ylabel_loss = 'MSE' + r'$\/(\frac{J}{m^2})$'
    xlabel = 'Epoch'
    title_train_loss = 'Val MSEloss vs Epoch on different layers'

    plt.title(title_train_loss, fontdict=title_font, weight='bold', pad=pad)

    for i in layers_list:
        result = pd.read_csv('./results/{}_layers.csv'.format(i))
        train_loss = np.array(eval(result.loc[0, 'val_loss']))
        plt.plot(tx, train_loss, label='{}_layers'.format(i))
    
    plt.legend(bbox_to_anchor=(num1, num2), title=None, loc=num3, fontsize=legend_size, ncol=1)
    plt.xlabel(xlabel, fontdict=label_font, weight='bold')
    plt.ylabel(ylabel_loss, fontdict=label_font, weight='bold')
    plt.xticks(xticks[1], fontsize=ticks_size, weight='bold')
    plt.yticks(yticks[1], fontsize=ticks_size, weight='bold')
    plt.tick_params(labelsize=ticks_size, length=5, width=1.4)
    tk = plt.gca()
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.subplots_adjust(top=0.89,
                    bottom=0.16,
                    left=0.15,
                    right=0.85,
                    hspace=0.2,
                    wspace=0.2)

    fig_train_loss.savefig("./images/val_loss_on_layers.png")


def plot_val_r2_neuron_number(layers_list):
    fig_train_loss = plt.figure(figsize=figure_size, dpi=dpi)

    xlim = (-10, 1000)
    ylim = (0.5, 1)
    xticks = (True, (0, 200, 400, 600, 800, 1000))
    yticks = (True, tuple([i for i in np.arange(0.5, 1.01, 0.1)]))
    num1, num2, num3, num4 = 0.99, 0.5, 1, 0
    tx = np.arange(0, epochs, 1)
    ylabel_r2 = r'$R^2$'
    xlabel = 'Epoch'
    title_train_loss = 'Validation R2 vs Epoch on different layers'

    plt.title(title_train_loss, fontdict=title_font, weight='bold', pad=pad)

    for i in layers_list:
        result = pd.read_csv('./results/{}_layers.csv'.format(i))
        train_loss = np.array(eval(result.loc[0, 'val_r2']))
        plt.plot(tx, train_loss, label='{}_layers'.format(i))
    
    plt.legend(bbox_to_anchor=(num1, num2), title=None, loc=num3, fontsize=legend_size, ncol=1)
    plt.xlabel(xlabel, fontdict=label_font, weight='bold')
    plt.ylabel(ylabel_r2, fontdict=label_font, weight='bold')
    plt.xticks(xticks[1], fontsize=ticks_size, weight='bold')
    plt.yticks(yticks[1], fontsize=ticks_size, weight='bold')
    plt.tick_params(labelsize=ticks_size, length=5, width=1.4)
    tk = plt.gca()
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.subplots_adjust(top=0.89,
                    bottom=0.16,
                    left=0.15,
                    right=0.85,
                    hspace=0.2,
                    wspace=0.2)

    fig_train_loss.savefig("./images/val_r2_on_layers.png")


def plot_train_loss_with_neuron_number():
    fig_train_loss = plt.figure(figsize=figure_size, dpi=dpi)

    xlim = (-10, 1000)
    ylim = (0, 0.01)
    xticks = (True, (0, 200, 400, 600, 800, 1000))
    yticks = (True, tuple([i for i in np.arange(0, 0.011, 0.002)]))
    num1, num2, num3, num4 = 0.99, 0.99, 1, 0          # 用于调整图例的位置，使其不遮挡图像。num1和num2分别是图例的x和y坐标，num3控制图例位置。
    tx = np.arange(0, epochs, 1)
    ylabel_loss = 'MSE' + r'$\/(\frac{J}{m^2})$'
    xlabel = 'Epoch'
    title_train_loss = 'Training MSEloss vs Epoch on different neuron_number'

    plt.title(title_train_loss, fontdict=title_font, weight='bold', pad=pad)

    for i in range(1, 21):
        result = pd.read_csv('../results/1500/{}_neuron.csv'.format(i))
        train_loss = np.array(eval(result.loc[0, 'train_loss']))
        plt.plot(tx, train_loss, label='{}_neuron'.format(i))
    
    plt.legend(bbox_to_anchor=(num1, num2), title=None, loc=num3, fontsize=legend_size, ncol=1)
    plt.xlabel(xlabel, fontdict=label_font, weight='bold')
    plt.ylabel(ylabel_loss, fontdict=label_font, weight='bold')
    plt.xticks(xticks[1], fontsize=ticks_size, weight='bold')
    plt.yticks(yticks[1], fontsize=ticks_size, weight='bold')
    plt.tick_params(labelsize=ticks_size, length=5, width=1.4)
    tk = plt.gca()
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.subplots_adjust(top=0.89,
                    bottom=0.16,
                    left=0.15,
                    right=0.85,
                    hspace=0.2,
                    wspace=0.2)

    fig_train_loss.savefig("./neuron_number_train_loss.png")


def plot_val_loss_with_neuron_number():
    fig_train_loss = plt.figure(figsize=figure_size, dpi=dpi)

    xlim = (-10, 1000)
    ylim = (0, 0.01)
    xticks = (True, (0, 200, 400, 600, 800, 1000))
    yticks = (True, tuple([i for i in np.arange(0, 0.011, 0.002)]))
    num1, num2, num3, num4 = 0.99, 0.99, 1, 0          # 用于调整图例的位置，使其不遮挡图像。num1和num2分别是图例的x和y坐标，num3控制图例位置。
    tx = np.arange(0, epochs, 1)
    ylabel_loss = 'MSE' + r'$\/(\frac{J}{m^2})$'
    xlabel = 'Epoch'
    title_train_loss = 'Validation MSEloss vs Epoch on different neuron_number'

    plt.title(title_train_loss, fontdict=title_font, weight='bold', pad=pad)

    for i in range(1, 21):
        result = pd.read_csv('../results/1500/{}_neuron.csv'.format(i))
        train_loss = np.array(eval(result.loc[0, 'val_loss']))
        plt.plot(tx, train_loss, label='{}_neuron'.format(i))
    
    plt.legend(bbox_to_anchor=(num1, num2), title=None, loc=num3, fontsize=legend_size, ncol=1)
    plt.xlabel(xlabel, fontdict=label_font, weight='bold')
    plt.ylabel(ylabel_loss, fontdict=label_font, weight='bold')
    plt.xticks(xticks[1], fontsize=ticks_size, weight='bold')
    plt.yticks(yticks[1], fontsize=ticks_size, weight='bold')
    plt.tick_params(labelsize=ticks_size, length=5, width=1.4)
    tk = plt.gca()
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.subplots_adjust(top=0.89,
                    bottom=0.16,
                    left=0.15,
                    right=0.85,
                    hspace=0.2,
                    wspace=0.2)

    fig_train_loss.savefig("./neuron_number_val_loss.png")


def plot_val_r2_with_neuron_number():
    fig_train_loss = plt.figure(figsize=figure_size, dpi=dpi)

    xlim = (-10, 1000)
    ylim = (0, 1)
    xticks = (True, (0, 200, 400, 600, 800, 1000))
    yticks = (True, tuple([i for i in np.arange(0, 1.01, 0.1)]))
    num1, num2, num3, num4 = 0.99, 0.99, 1, 0          # 用于调整图例的位置，使其不遮挡图像。num1和num2分别是图例的x和y坐标，num3控制图例位置。
    tx = np.arange(0, epochs, 1)
    ylabel_loss = r'$R^2$'
    xlabel = 'Epoch'
    title_train_loss = 'Validation R2 vs Epoch on different neuron_number'

    plt.title(title_train_loss, fontdict=title_font, weight='bold', pad=pad)

    for i in range(1, 21):
        result = pd.read_csv('../results/1500/{}_neuron.csv'.format(i))
        train_loss = np.array(eval(result.loc[0, 'val_loss']))
        plt.plot(tx, train_loss, label='{}_neuron'.format(i))
    
    plt.legend(bbox_to_anchor=(num1, num2), title=None, loc=num3, fontsize=legend_size, ncol=1)
    plt.xlabel(xlabel, fontdict=label_font, weight='bold')
    plt.ylabel(ylabel_loss, fontdict=label_font, weight='bold')
    plt.xticks(xticks[1], fontsize=ticks_size, weight='bold')
    plt.yticks(yticks[1], fontsize=ticks_size, weight='bold')
    plt.tick_params(labelsize=ticks_size, length=5, width=1.4)
    tk = plt.gca()
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.subplots_adjust(top=0.89,
                    bottom=0.16,
                    left=0.15,
                    right=0.85,
                    hspace=0.2,
                    wspace=0.2)

    fig_train_loss.savefig("./neuron_number_val_r2.png")


if __name__ == "__main__":
    feature_list = ['VEC mean', 'APE mean', 'Electronegativity local mismatch',
       'Shear modulus delta', 'Shear modulus mean', 'Mixing enthalpy',
       'Shear modulus strength model', 'Nb', 'Mo', 'Radii gamma',
       'Yang omega', 'Electronegativity delta', 'Lambda entropy', 'Ta',
       'Shear modulus local mismatch', 'Configuration entropy',
       'Yang delta', 'W', 'Radii local mismatch', 'Mean cohesive energy',
       'Total weight', 'Atomic weight mean']
    # plot_train_loss_with_feature_number(feature_list=feature_list)
    # plot_val_loss_with_feature_number(feature_list=feature_list)
    # plot_val_r2_with_feature_number(feature_list=feature_list)
    # plot_val_loss_neuron_number(layers_list=[1, 2, 3, 4])
    # plot_val_loss_neuron_number(layers_list=[1, 2, 3, 4])
    # plot_val_r2_neuron_number(layers_list=[1, 2, 3, 4])
    plot_loss('L2')