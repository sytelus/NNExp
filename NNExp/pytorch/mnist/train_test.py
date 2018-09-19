import torch
import torch.optim as optim
import torch.nn.functional as F
import utils

class TrainTest:
    def __init__(self, model, lr, momentum, train_device, test_device):
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.optimizer = optim.SGD(model.parameters(), lr, momentum)
        self.train_device = train_device
        self.test_device = test_device

    def train_epoch(self, train_loader, track_train_accuracy = True):
        self.model.to(self.train_device)
        train_loss_sum = 0
        correct = 0
        self.model.train()

        for input, label in train_loader:
            input, label = input.to(self.train_device), label.to(self.train_device)
            self.optimizer.zero_grad()
            output = self.model(input)
            loss = F.nll_loss(output, label)
            loss.backward()
            self.optimizer.step()
            train_loss_sum += loss.item()

            if track_train_accuracy:
                self.model.eval()
                with torch.no_grad():
                    output = self.model(input)
                    pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                    correct += pred.eq(label.view_as(pred)).sum().item()
                self.model.train()
        self.train_loss = train_loss_sum / len(train_loader)
        self.train_accuracy = correct / len(train_loader.dataset)

    def test_epoch(self, test_loader):
        self.model.to(self.test_device)
        self.model.eval()
        test_loss_sum = 0
        correct = 0
        with torch.no_grad():
            for input, label in test_loader:
                input, label = input.to(self.test_device), label.to(self.test_device)
                output = self.model(input)
                test_loss_sum += F.nll_loss(output, label, reduction = 'sum').item() # sum up batch loss
                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(label.view_as(pred)).sum().item()
        self.test_loss = test_loss_sum / len(test_loader.dataset)
        self.test_accuracy = correct / len(test_loader.dataset)

    def train_model(self, epochs, train_loader, test_loader, epoch_callback):
        for epoch in range(0, epochs):
            with utils.MeasureBlockTime(no_print=True) as train_time:
                self.train_epoch(train_loader)
            with utils.MeasureBlockTime(no_print=True) as test_time:
                self.test_epoch(test_loader)
            epoch_callback(self, epoch, train_time.elapsed, test_time.elapsed)