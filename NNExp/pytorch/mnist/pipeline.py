from __future__ import print_function
import argparse
import torch
from mnist_conv import MnistConv as Net
from torchvision import datasets, transforms
import utils
from data_tools import DataTools
from train_test import TrainTest
from debug_probe import DebugProbe

def getArgs():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', #64
                        help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', #0.01
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum')
    parser.add_argument('--no-cuda-train', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-cuda-test', action='store_true', default=False,
                        help='disables CUDA testing')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    args = parser.parse_args()

    args.use_cuda_train = not args.no_cuda_train and torch.cuda.is_available()
    args.use_cuda_test = not args.no_cuda_test and torch.cuda.is_available()
    args.device_train = torch.device("cuda" if args.use_cuda_train else "cpu")
    args.device_test = torch.device("cuda" if args.use_cuda_test else "cpu")
    args.data_per_class = None
    return args

def main():
    #prepare args
    args = getArgs()
    print("Train CUDA: {}, Test CUDA: {}, Batch Size: {}".format(args.use_cuda_train, args.use_cuda_test, args.batch_size))

    #init stuff
    torch.manual_seed(args.seed)
    model = Net()

    #load data
    with utils.MeasureBlockTime("Data loading (s): "):
        train_loader, test_loader = DataTools.getDataLoaders(args)
    print("Train: {}, Test:{}, Train batches: {}, Test batches:{}".format(
        len(train_loader.dataset), len(test_loader.dataset), len(train_loader), len(test_loader)))
    train_test = TrainTest(model, args.lr, args.momentum, args.device_train, args.device_test)    
    probe = DebugProbe(train_test, "test_exp")

    #train model
    train_test.train_model(args.epochs, train_loader, test_loader)


if __name__ == '__main__':
    main()