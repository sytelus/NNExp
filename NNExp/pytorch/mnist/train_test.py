import torch
import torch.optim as optim
import torch.nn.functional as F
import utils

class TrainTest:
    class Callbacks:
        after_epoch = lambda epoch, train_time, test_time: None
        after_train_batch = lambda model, input, label, output, pred, loss, correct:None
        after_first_train_batch = lambda input, label, output, pred, loss, correct:None

    def __init__(self, model, lr, momentum, train_device, test_device):
        self.train_device = train_device
        self.test_device = test_device
        self.lr = lr
        self.momentum = momentum
        self.callbacks = TrainTest.Callbacks()

        self.model = model
        if self.train_device == self.test_device:
            self.model.to(self.train_device)
        self.optimizer = optim.SGD(model.parameters(), lr, momentum)
        self.is_first_batch = False


    def train_epoch(self, train_loader, track_train_accuracy = True):
        if self.train_device != self.test_device:
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

            if track_train_accuracy:
                self.model.eval() # disable layers like dropout
                with torch.no_grad():
                    output = self.model(input)
                    train_loss_sum += F.nll_loss(output, label, reduction = 'sum').item() # sum up batch loss
                    pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                    correct += pred.eq(label.view_as(pred)).sum().item()
                self.model.train()
            else:
                train_loss_sum += loss.item() * len(input) #this won't be accurate because of layers like dropout

            self.callbacks.after_train_batch(input, label, output, pred, loss, correct)
            if not self.is_first_batch:
                self.callbacks.after_first_train_batch(input, label, output, pred, loss, correct)
                self.is_first_batch = True

        self.train_loss = train_loss_sum / len(train_loader.dataset)
        self.train_accuracy = correct / len(train_loader.dataset)

    def test_epoch(self, test_loader):
        if self.train_device != self.test_device:
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

    def train_model(self, epochs, train_loader, test_loader):
        for epoch in range(0, epochs):
            with utils.MeasureBlockTime(no_print=True) as train_time:
                self.train_epoch(train_loader)
            with utils.MeasureBlockTime(no_print=True) as test_time:
                self.test_epoch(test_loader)
            self.callbacks.after_epoch(epoch, train_time.elapsed, test_time.elapsed)