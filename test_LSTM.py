# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 15:55:54 2024

@author: zhua079
"""
import os
import torch.nn as nn
import torch.utils.data as Data
from joblib import dump, load
import torch
import numpy as np
import pandas as pd
from scipy.io import loadmat
import torch.optim as optim
from matrix_aug import *
from matrixdatasets import dataset
from sklearn.model_selection import train_test_split
from CWRUdata import get_files as CWRU
from datetime import datetime

data_dir = r'D:\Downloads\Mechanical-datasets-master\dataset'
list_data1 = CWRU(data_dir, [2])
data, label = list_data1[0], list_data1[1]

# 参数与配置

torch.manual_seed(0)  # 设置随机种子，以使实验结果具有可重复性
device = torch.device("cuda" if torch.cuda.is_available()
                      else "cpu")  # 有GPU先用GPU训练


class VariableSizeLSTM(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(VariableSizeLSTM, self).__init__()

        # Create a list of LSTM layers with different hidden sizes
        self.lstms = nn.ModuleList([
            nn.LSTMCell(input_size, hidden_sizes[0]),
            nn.LSTMCell(hidden_sizes[0], hidden_sizes[1]),
            nn.LSTMCell(hidden_sizes[1], hidden_sizes[2])
        ])

        self.fc = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        # Initialize cell and hidden states
        h_t = [torch.zeros(x.size(0), size).to(x.device) for size in [
            hidden_sizes[0], hidden_sizes[1], hidden_sizes[2]]]
        c_t = [torch.zeros(x.size(0), size).to(x.device) for size in [
            hidden_sizes[0], hidden_sizes[1], hidden_sizes[2]]]

        for i in range(x.size(1)):
            h_t[0], c_t[0] = self.lstms[0](x[:, i, :], (h_t[0], c_t[0]))
            h_t[1], c_t[1] = self.lstms[1](h_t[0], (h_t[1], c_t[1]))
            h_t[2], c_t[2] = self.lstms[2](h_t[1], (h_t[2], c_t[2]))

        out = self.fc(h_t[2])
        return out


batch_size = 32


def data_transforms(dataset_type="train", normlize_type="-1-1"):
    transforms = {
        'train': Compose([
            # Reshape(32, 32)
        ]),
        'test': Compose([
            # Reshape(32, 32)
        ])
    }
    return transforms[dataset_type]


data_pd = pd.DataFrame({"data": data, "label": label})
train_pd, test_pd = train_test_split(data_pd, test_size=0.2)


normlizetype = "mean-std"
train_dataset = dataset(
    list_data=train_pd, transform=data_transforms('train', normlizetype))
test_dataset = dataset(
    list_data=test_pd, transform=data_transforms('test', normlizetype))

train_num = len(train_pd)
test_num = len(test_pd)

datasets = {}
datasets['train'], datasets['test'] = train_dataset, test_dataset

dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size,
                                              shuffle=(
    True if x == 'train' else False),
    num_workers=0,
    pin_memory=(True if device == 'cuda' else False))
    for x in ['train', 'test']}

input_size = 32
hidden_sizes = [64, 64, 128]  # Set the hidden sizes for each layer
output_size = 10
sequence_length = 32


# Create an instance of the variable size LSTM model
net = VariableSizeLSTM(
    input_size, hidden_sizes, output_size)

net.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)


best_acc = 0.65

'''
model save path

'''
sub_dir = 'CWRU_' + datetime.strftime(datetime.now(), '%y-%m-%d-%H%M')
save_dir = os.path.join(r'D:\Github\Machine learning\results', sub_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

epochs = 100

acc = []
for epoch in range(epochs):
    running_loss = 0.0
    correct_train = 0
    epoch_acc = 0
    for phase in ['train', 'test']:

        if phase == 'train':
            net.train()
            for train_steps, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.reshape(inputs.size()[0], 32, 32).to(
                    torch.float).to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = net(inputs)
                pred_train = torch.max(outputs, dim=1)[1]
                correct_train += torch.eq(pred_train,
                                          labels).float().sum().item()

                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            train_acc = correct_train / train_num
            acc.append(train_acc)
            print('[epoch %d]  train_acc: %.3f' %
                  (epoch + 1,  train_acc))

        else:
            net.eval()
            with torch.no_grad():
                if phase == 'test':
                    correct_test = 0.0
                    # src_val_bar = tqdm(dataloaders[phase], file=sys.stdout)
                    # for steps, (inputs, labels) in enumerate(src_val_bar):
                    for steps, (test_inputs, test_labels) in enumerate(dataloaders[phase]):
                        test_inputs = test_inputs.reshape(
                            test_inputs.size()[0], 32, 32).to(torch.float).to(device)
                        test_labels = test_labels.to(device)

                        outputs = net(test_inputs)
                        pred_test_labels = torch.max(outputs, dim=1)[1]
                        correct_test += torch.eq(pred_test_labels,
                                                 test_labels).float().sum().item()

                    test_acc = correct_test / test_num
                    epoch_acc = test_acc
                    acc.append(test_acc)
                    # src_val_bar.desc = "src_val epoch[{}/{}] loss:{:.3f} acc:{:.3f}".format(epoch+1, epochs,
                    #                                                                     running_loss,
                    #                                                                    src_train_acc)

                    print('[epoch %d]  test_acc: %.3f' %
                          (epoch + 1,  test_acc))

                    if epoch_acc > best_acc or epoch == epochs:
                        best_acc = epoch_acc
                        save_path = save_dir + \
                            '\epoch-{}_{:.3f}.pth'.format(epoch, best_acc)
                        torch.save(net.state_dict(), save_path)

print('Finished Training')
