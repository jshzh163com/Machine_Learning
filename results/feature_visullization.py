# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 15:43:06 2023

@author: zhua079
"""
from matplotlib import pyplot as plt
from sklearn import manifold
import random
from datasets.multi_UO import get_files
import numpy as np
from models.cnn_1d import cnn_features as cnn_features_1d
import torch
from torch import nn
import datasets
from torch.utils.data import DataLoader
import imageio.v2 as imageio

#   file path
data_dir = r'D:\Downloads\uottawa_Huang'


'''
model
'''
bottleneck_num = 512
cnn = cnn_features_1d()
bottleneck_layer = nn.Sequential(nn.Linear(cnn.output_num(), bottleneck_num),
                                 nn.ReLU(inplace=True), nn.Dropout())
classifier_layer = nn.Linear(bottleneck_num, 5)
model = nn.Sequential(cnn, bottleneck_layer, classifier_layer)


def set_random_seed(seed=0):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_random_seed(0)
'''
dataloader
'''
device = torch.device("cuda")
Dataset = getattr(datasets, 'multi_UO')
datasets = {}
datasets['source_train'], datasets['source_val'], datasets['target_train'], datasets['target_val'] = Dataset(
    data_dir, [[0, 1, 2], [3]], 'mean-std').data_split(transfer_learning=True)

dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=32,  # args.batch_size
                                              shuffle=(True),
                                              num_workers=0,
                                              pin_memory=(
    True if device == torch.device("cuda") else False),
    drop_last=(True))
    for x in ['source_train', 'source_val', 'target_train', 'target_val']}

# load_model_dir = r'D:\Github\UDTL_multi_domain-main\checkpoint\DA\s_0_1_2_t3\cnn_features_1d_1116-230439\96-0.2552-best_model.pth' # worse 5 classes
# load_model_dir = r'D:\Github\UDTL_multi_domain-main\checkpoint\DA\s_0_1_2_t3\cnn_features_1d_1116-230157\93-0.9635-best_model.pth'  # accurate one


def main(iter):
    load_model_path = r'D:\Github\UDTL_multi_domain-main\o\\'
    load_model_dir = load_model_path + str(iter) + '.pth'
    model.load_state_dict(torch.load(load_model_dir))

    model1 = nn.Sequential(*list(model.children())[:-1])

    model1.to(torch.device("cuda"))

    tar_val_features = []
    tar_val_labels = []
    with torch.no_grad():
        for batch_data, batch_labels in dataloaders['target_val']:
            # 将数据输入模型
            output_features = model1(batch_data.to(torch.device("cuda")))

            # 保存特征和标签
            tar_val_features.append(output_features.cpu().numpy())
            tar_val_labels.append(batch_labels.cpu().numpy())

    src_train_features = []
    src_train_labels = []
    with torch.no_grad():
        for batch_data, batch_labels in dataloaders['source_train']:
            # 将数据输入模型
            output_features = model1(batch_data.to(torch.device("cuda")))

            # 保存特征和标签
            src_train_features.append(output_features.cpu().numpy())
            src_train_labels.append(batch_labels.cpu().numpy())

    src_val_features = []
    src_val_labels = []
    with torch.no_grad():
        for batch_data, batch_labels in dataloaders['source_val']:
            # 将数据输入模型
            output_features = model1(batch_data.to(torch.device("cuda")))

            # 保存特征和标签
            src_val_features.append(output_features.cpu().numpy())
            src_val_labels.append(batch_labels.cpu().numpy())

    src_val = np.concatenate(src_val_features)
    src_val_label = np.concatenate(src_val_labels)
    tar_val = np.concatenate(tar_val_features)
    tar_val_label = np.concatenate(tar_val_labels)

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=20)
    X = np.vstack((src_val, tar_val))
    X_tsne = tsne.fit_transform(X)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)

    label0 = src_val_label
    label1 = tar_val_label

    font1 = {'family': 'Times New Roman',
             'size': '14'}
    # fig = plt.figure()
    plt.scatter(X_norm[:len(label0), 0],
                X_norm[:len(label0), 1], c=label0, label=label0)
    plt.scatter(X_norm[len(label0):, 0], X_norm[len(label0):, 1],
                c=label1, marker='*', label=label1)
    # plt.legend(handles = plot.legend_elements()[0], labels = list(target_names), prop=font1)
    plt.title(f'Epoch: {iter}', font1)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks([])
    plt.yticks([])

    plt.savefig('temp.png')

    writer.append_data(imageio.imread('temp.png'))  # 读取并添加图形到 GIF
    plt.clf()


if __name__ == '__main__':

    lists = [3*iter + 1 for iter in range(34)]
    writer = imageio.get_writer(
        'feature_visualization.gif', duration=0.8)
    for iter in lists:
        main(iter)
    writer.close()
    plt.show()
    # plt.savefig('UO_feature_visualization_t-SNE.png',
    #             bbox_inches='tight', dpi=600)
