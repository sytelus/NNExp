from __future__ import print_function
import argparse
import torch
from mnist_cwconv import MnistCwConv
from torchvision import datasets, transforms
import utils
from data_tools import DataTools
from tensorboardX import SummaryWriter
from train_test import TrainTest

def getArgs():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N', #64
                        help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', #0.01
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum')
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
    writer = SummaryWriter('/tlogs/exp1')
    use_cuda_train = not args.no_cuda_train and torch.cuda.is_available()
    use_cuda_test = not args.no_cuda_test and torch.cuda.is_available()
    print("Train CUDA: {}, Test CUDA: {}, Batch Size: {}".format(use_cuda_train, use_cuda_test, args.batch_size))
    torch.manual_seed(args.seed)
    device_train = torch.device("cuda" if use_cuda_train else "cpu")
    device_test = torch.device("cuda" if use_cuda_test else "cpu")
    kwargs_train ={'num_workers': 1, 'pin_memory': True} if use_cuda_train else {}
    kwargs_test ={'num_workers': 1, 'pin_memory': True} if use_cuda_test else {}

    with utils.MeasureBlockTime("Data loading (s): "):
        train_loader, test_loader = DataTools.getDataLoaders(args.batch_size, args.test_batch_size, 10, kwargs_train, kwargs_test)
    print("Train: {}, Test:{}".format(len(train_loader), len(test_loader)))
    model = MnistCwConv()
    train_test = TrainTest(model, args.lr, args.momentum, device_train, device_test)

    for epoch in range(1, args.epochs + 1):
        with utils.MeasureBlockTime(no_print=True) as train_time:
            train_test.train(train_loader)
        with utils.MeasureBlockTime(no_print=True) as test_time:
            train_test.test(test_loader)
        writer.add_scalar('losses/train', train_test.train_loss, epoch)
        writer.add_scalar('losses/test', train_test.test_loss, epoch)
        writer.add_scalar('accuracy/test', train_test.accuracy, epoch)

        print("Epoch: {}, train_loss: {:.2f}, test_loss: {:.2f}, accuracy:{:.2f}, TrainTime: {:.2f}, , TestTime: {:.2f}".format(
            epoch, train_test.train_loss, train_test.test_loss, 
            train_test.accuracy, train_time.elapsed, test_time.elapsed))


if __name__ == '__main__':
    main()