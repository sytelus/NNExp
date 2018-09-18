import torch
import torch.optim as optim
import torch.nn.functional as F

class TrainTest:
    def __init__(self, model, lr, momentum, train_device, test_device):
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.optimizer = optim.SGD(model.parameters(), lr, momentum)
        self.train_device = train_device
        self.test_device = test_device

    def train(self, train_loader):
        self.model.to(self.train_device)
        self.model.train()
        loss = torch.tensor(-1, dtype=torch.double)
        for batch_idx, (input, label) in enumerate(train_loader):
            input, label = input.to(self.train_device), label.to(self.train_device)
            self.optimizer.zero_grad()
            output = self.model(input)
            self.loss = F.nll_loss(output, label)
            self.loss.backward()
            self.optimizer.step()
        self.train_loss = loss.item()

    def test(self, test_loader):
        self.model.to(self.test_device)
        self.model.eval()
        test_loss_sum = 0
        self.correct = 0
        with torch.no_grad():
            for input, label in test_loader:
                input, label = input.to(self.test_device), label.to(self.test_device)
                output = self.model(input)
                test_loss_sum += F.nll_loss(output, label, reduction = 'sum').item() # sum up batch loss
                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                self.correct += pred.eq(label.view_as(pred)).sum().item()
        self.test_len = len(test_loader.dataset)
        self.test_loss = test_loss_sum / self.test_len
        self.accuracy = self.correct / self.test_len