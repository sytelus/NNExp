import torch
import numpy as np

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


def getNorm(dl):
    avg = None
    var = None
    l = [data for data, _ in dl]
    l = torch.cat(l, dim=0) #[60000, 1, 28, 28])
    l = torch.transpose(l, 0, 1).contiguous()
    l = l.view(l.size(0), -1)
    mean = torch.mean(l, dim=1)
    std = torch.std(l, dim=1)
    print(mean)

def main():
    train_ds = datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
    train_dl = torch.utils.data.DataLoader(train_ds)

    getNorm(train_dl)

if __name__ == '__main__':
    main()