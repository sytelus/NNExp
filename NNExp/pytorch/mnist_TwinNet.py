from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import win_unicode_console

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.optimizer = optim.SGD(self.parameters(), lr=args.lr, momentum=args.momentum)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_decay_epochs, gamma=args.lr_decay_gamma)

    def forward_fc(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

    def forward(self, x):
        x = self.forward_fc(x)
        return F.log_softmax(x, dim=1)

class TwinNet(Net):
    def __init__(self, args):
        super(TwinNet, self).__init__(args)
        self.net1 = Net(args)
        self.net2 = Net(args)
        self.optimizer = optim.SGD(self.parameters(), lr=args.lr, momentum=args.momentum)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_decay_epochs, gamma=args.lr_decay_gamma)
        
    def forward(self, x):
        x = (self.net1.forward_fc(x) + self.net2.forward_fc(x)) / 2.0
        return F.log_softmax(x, dim=1)

def train(args, model, device, train_loader, epoch):
    model.train()
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        model.optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        model.optimizer.step()
    
    model.scheduler.step()

        #if i % args.log_interval == 0:
        #    print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
        #        epoch, i * len(data), len(train_loader.dataset),
        #        100. * i / len(train_loader), loss.item()))

def cotrain(args, model, device, train_loader, net, epoch):
    model.train()
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        model.optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        net.optimizer.step()
    net.scheduler.step()
        #if i % args.log_interval == 0:
        #    print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
        #        epoch, i * len(data), len(train_loader.dataset),
        #        100. * i / len(train_loader), loss.item()))

def cotrain2(args, device, model, train_loader1, train_loader2, epoch):
    model.train()
    net1 = model.net1
    net2 = model.net2

    for i, (d1, d2) in enumerate(zip(train_loader1, train_loader2)):
        data1, target1 = d1[0].to(device), d1[1].to(device)
        data2, target2 = d2[0].to(device), d2[1].to(device)

        model.optimizer.zero_grad()
        output = model(data1)
        loss = F.nll_loss(output, target1)
        loss.backward()
        net1.optimizer.step()

        model.optimizer.zero_grad()
        output = model(data2)
        loss = F.nll_loss(output, target2)
        loss.backward()
        net2.optimizer.step()

    net1.scheduler.step()
    net2.scheduler.step()

        #if i % args.log_interval == 0:
        #    print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
        #        epoch, i * len(data), len(train_loader.dataset),
        #        100. * i / len(train_loader), loss.item()))

def test(args, model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    try:
        print('Test Epoch: {} Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            epoch, test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    except OSError as e:
        pass

def parseArgs():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train (default: 500)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--lr-decay-gamma', type=int, default=0.1,
                        help='step decay rate for lr-decay-epochs epochs')
    parser.add_argument('--lr-decay-epochs', type=int, default=1000,
                        help='step decay rate after these epochs')
    args = parser.parse_args()
    return args

def getDatasets(args, use_cuda):
    print("Loading data...")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_db = list(datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])))
    test_db = list(datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])))

    print("Shuffling...")
    np.random.shuffle(train_db)

    print("Prepping loaders...")
    train_db1 = train_db[::200]
    train_db2 = train_db[1::200]
    train_loader1 = torch.utils.data.DataLoader(train_db1,
        batch_size=args.batch_size, shuffle=True, **kwargs)
    train_loader2 = torch.utils.data.DataLoader(train_db2,
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_db,
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    print("Data lens: set 1 size {}, set 2 size {}".format(len(train_db1), len(train_db2)))

    print("Data loading done.")

    return test_loader, train_loader1, train_loader2

def main():
    win_unicode_console.enable()

    # Training settings
    args = parseArgs()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    test_loader, train_loader1, train_loader2 = getDatasets(args, use_cuda)

    model = TwinNet(args).to(device)

    for epoch in range(1, args.epochs + 1):
        #train(args, model.net1, device, train_loader1, epoch)
        #train(args, model.net2, device, train_loader2, epoch)
        #train(args, model.net1, device, train_loader1, epoch)
        #train(args, model.net2, device, train_loader2, epoch)
        #cotrain(args, model, device, train_loader1, model.net1, epoch)
        #cotrain(args, model, device, train_loader2, model.net2, epoch)
        cotrain2(args, device, model, train_loader1, train_loader2, epoch)
        test(args, model.net1, device, test_loader, epoch)
        test(args, model.net2, device, test_loader, epoch)


if __name__ == '__main__':
    main()
