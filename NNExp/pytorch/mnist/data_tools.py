import torch
import numpy as np

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler

class DataTools:
    def get1ChannelNorm(ds):
        l = [data for data, _ in ds]
        l = torch.cat(l, dim=0) #size: [60000, 28, 28]
        l = l.view(1, -1) #size: [1, 47040000]
        mean = torch.mean(l, dim=1)
        std = torch.std(l, dim=1)
        retun (mean, std) # (tensor([0.1305]), tensor([0.3081])

    def getNChannelNorm(ds):
        l = [data for data, _ in ds]
        l = torch.cat(l, dim=0) #size: [N, 3, X, Y]
        l = torch.transpose(l, 0, 1).contiguous() #size: [3, N, X, Y]
        l = l.view(l.size(0), -1) #size: [N, [R, G, B]]
        mean = torch.mean(l, dim=1) #size: [R, G, B]
        std = torch.std(l, dim=1) #size: [R, G, B]
        retun (mean, std)

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
                train_data.append(data)
                train_label.append(torch.unsqueeze(label, 0))
            else:
                test_data.append(data)
                test_label.append(torch.unsqueeze(label, 0))
        train_data = torch.cat(train_data)
        for ll in train_label:
            print(ll)
        train_label = torch.cat(train_label)
        test_data = torch.cat(test_data)
        test_label = torch.cat(test_label)

        return (TensorDataset(train_data, train_label), 
            TensorDataset(test_data, test_label))

def getPartialDataSets():
    train_ds = datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
    train_ds, test_ds = sampleFromClass(train_ds, 3)
    return train_ds, test_ds

def main():
    #get1ChannelNorm(train_ds)


if __name__ == '__main__':
    main()