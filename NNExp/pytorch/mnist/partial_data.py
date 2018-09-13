from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import utils
from data_tools import DataTools
from cwlayers import CWConv2d, CWLinear


class Net(nn.Module):
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = CWConv2d(1, 10, kernel_size=5)
        self.conv2 = CWConv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = CWLinear(320, 50)
        self.fc2 = CWLinear(50, 10)
        self._initialize_weights()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    loss = torch.tensor(-1, dtype=torch.double)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
    return loss.item()

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction = 'sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_len = len(test_loader.dataset)
    return (test_loss / test_len, correct /test_len)

def getArgs():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda-train', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--no-cuda-test', action='store_true', default=False,
                        help='disables CUDA testing')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    return parser.parse_args()

def main():
    args = getArgs()
    use_cuda_train = not args.no_cuda_train and torch.cuda.is_available()
    use_cuda_test = not args.no_cuda_test and torch.cuda.is_available()
    print("Train CUDA: {}, Test CUDA: {}, Batch Size: {}".format(use_cuda_train, use_cuda_test, args.batch_size))
    torch.manual_seed(args.seed)
    device_train = torch.device("cuda" if use_cuda_train else "cpu")
    device_test = torch.device("cuda" if use_cuda_test else "cpu")
    kwargs_train = {'num_workers': 1, 'pin_memory': True} if use_cuda_train else {}
    kwargs_test = {'num_workers': 1, 'pin_memory': True} if use_cuda_test else {}

    with utils.MeasureBlockTime("Data loading (s): "):
        train_loader, test_loader = DataTools.getDataLoaders(args.batch_size, args.test_batch_size, 3, kwargs_train, kwargs_test)
    print("Train: {}, Test:{}".format(len(train_loader), len(test_loader)))
    model_train = Net().to(device_train)
    optimizer = optim.SGD(model_train.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        model_train = model_train.to(device_train)
        with utils.MeasureBlockTime(no_print=True) as train_time:
            train_loss = train(args, model_train, device_train, train_loader, optimizer, epoch)
        model_test = model_train.to(device_test)
        with utils.MeasureBlockTime(no_print=True) as test_time:
            test_loss, accuracy = test(args, model_test, device_test, test_loader)
        print("Epoch: {}, train_loss: {:.2f}, test_loss: {:.2f}, accuracy:{:.2f}, TrainTime: {:.2f}, , TestTime: {:.2f}".format(
            epoch, train_loss, test_loss, accuracy, train_time.elapsed, test_time.elapsed))


if __name__ == '__main__':
    main()