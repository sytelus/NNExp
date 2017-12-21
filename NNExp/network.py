import numpy as np
import network_config
import labeled_data

class Network:
    def __init__(self, config : network_config.NetworkConfig):
        self.config = config

    def feed_forward(self, input_vals):
        for b, w in zip(self.biases, self.weights):
            input_vals = self.config.activation_c.fn(np.dot(w, input_vals) + b)
        return input_vals

    def sgd(self, labeled_data : labeled_data.LabeledData):
        self.biases = self.config.init_c.get_biases(self.config.neuron_counts)
        self.weights = self.config.init_c.get_weights(self.config.neuron_counts);

        n = len(labeled_data.train)

        for j in range(self.config.epochs):
            np.random.shuffle(labeled_data.train)
            mini_batches = [
                labeled_data.train[k : k + self.config.batch_size]
                for k in range(0, n, self.config.batch_size)]
            for batch in mini_batches:
                self.sgd_batch(batch, len(labeled_data.train))
            print("Epoch %d : %d / %d" % (j, self.evaluate(labeled_data.test), len(labeled_data.test)))

    def sgd_batch(self, batch, n):
        b_batch = [np.zeros(b.shape) for b in self.biases]
        w_batch = [np.zeros(w.shape) for w in self.weights]

        for x, y_true, id in batch:
            b_backprop, w_backprop = self.config.backprop_c.fn(self.config, self.biases, self.weights, x, y_true)
            b_batch = [bb + bbp for bb, bbp in zip(b_batch, b_backprop)]
            w_batch = [wb + wbp for wb, wbp in zip(w_batch, w_backprop)]

        regularization = 1 - self.config.eta*(self.config.lmbda/n)
        eta_batch = self.config.eta / len(batch)

        self.weights = [regularization * w - eta_batch * wb
                        for w, wb in zip(self.weights, w_batch)]
        self.biases = [b - eta_batch * bb
                       for b, bb in zip(self.biases, b_batch)]


    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feed_forward(x)), np.argmax(y_true))
                        for (x, y_true, id) in test_data]
        return sum(int(y_pred == y_true) for (y_pred, y_true) in test_results)

