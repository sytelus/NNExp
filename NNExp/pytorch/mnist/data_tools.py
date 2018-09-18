import torch
import numpy as np

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler

class DataTools:
    @staticmethod
    def get1ChannelNorm(ds):
        l = [data for data, _ in ds]
        l = torch.cat(l, dim=0) #size: [60000, 28, 28]
        l = l.view(1, -1) #size: [1, 47040000]
        mean = torch.mean(l, dim=1)
        std = torch.std(l, dim=1)
        return (mean, std) # (tensor([0.1305]), tensor([0.3081])

    @staticmethod
    def getNChannelNorm(ds):
        l = [data for data, _ in ds]
        l = torch.cat(l, dim=0) #size: [N, 3, X, Y]
        l = torch.transpose(l, 0, 1).contiguous() #size: [3, N, X, Y]
        l = l.view(l.size(0), -1) #size: [N, [R, G, B]]
        mean = torch.mean(l, dim=1) #size: [R, G, B]
        std = torch.std(l, dim=1) #size: [R, G, B]
        return (mean, std)

    @staticmethod
    def sampleFromClass(ds, k):
        class_counts = {}
        train_data = []
        train_label = []
        test_data = []
        test_label = []
        for data, label in ds:
            c = label.item()
            class_counts[c] = class_counts.get(c, 0) + 1
            if class_counts[c] <= k:
                train_data.append(torch.unsqueeze(data, 0))
                train_label.append(torch.unsqueeze(label, 0))
            else:
                test_data.append(torch.unsqueeze(data, 0))
                test_label.append(torch.unsqueeze(label, 0))
        train_data = torch.cat(train_data)
        train_label = torch.cat(train_label)
        test_data = torch.cat(test_data)
        test_label = torch.cat(test_label)

        return (TensorDataset(train_data, train_label), 
            TensorDataset(test_data, test_label))

    @staticmethod
    def getFullDataSets():
        train_ds = datasets.MNIST('../data', train=True, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor() #,
                                #transforms.Normalize((0.1307,), (0.3081,))
                        ]));
        test_ds = datasets.MNIST('../data', train=False, transform=transforms.Compose([
                               transforms.ToTensor() #,
                               #transforms.Normalize((0.1307,), (0.3081,))
                           ]));
        return train_ds, test_ds

    @staticmethod
    def getDataSets(k=None):
        train_ds, test_ds = DataTools.getFullDataSets()
    
        if k is not None:
            train_ds_part, test_ds_part = DataTools.sampleFromClass(train_ds, k)
            test_ds_part = test_ds
        else:
            train_ds_part, test_ds_part = train_ds, test_ds

        return train_ds_part, test_ds_part

    @staticmethod
    def getDataLoaders(batch_size, test_batch_size, k=None, kwargs_train={}, kwargs_test={}):
        train_ds, test_ds = DataTools.getDataSets(k)

        train_loader = torch.utils.data.DataLoader(train_ds,
            batch_size=batch_size, shuffle=True, **kwargs_train)

        test_loader = torch.utils.data.DataLoader(test_ds,
            batch_size=test_batch_size, shuffle=True, **kwargs_test)

        return train_loader, test_loader