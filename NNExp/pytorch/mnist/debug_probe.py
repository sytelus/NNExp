from tensorboardX import SummaryWriter

class DebugProbe:
    def __init__(self, train_test, exp_name, log_dir='d:/tlogs/'):
        self.train_writer = SummaryWriter(log_dir + exp_name + '/train/')
        self.test_writer = SummaryWriter(log_dir + exp_name + '/test/')
        self.train_test = train_test 

        train_test.callbacks.after_epoch = lambda epoch, train_time, test_time: self.log_epoch(epoch, train_time, test_time)
        train_test.callbacks.after_first_train_batch = lambda input, label, output, pred, loss, correct: self.log_model(input)

    def log_epoch(self, epoch, train_time, test_time):
        self.train_writer.add_scalar('loss', self.train_test.train_loss, epoch)
        self.test_writer.add_scalar('loss', self.train_test.test_loss, epoch)
        self.train_writer.add_scalar('accuracy', self.train_test.train_accuracy, epoch)
        self.test_writer.add_scalar('accuracy', self.train_test.test_accuracy, epoch)

        print("Epoch: {}, train_loss: {:.2f}, train_accuracy: {:.4f}, test_loss: {:.2f}, test_accuracy:{:.4f}, TrainTime: {:.2f}, , TestTime: {:.2f}".format(
            epoch, self.train_test.train_loss, self.train_test.train_accuracy, self.train_test.test_loss, self.train_test.test_accuracy, train_time, test_time))

    def log_model(self, input):
        self.test_writer.add_graph(self.train_test.model, input, verbose = True)



