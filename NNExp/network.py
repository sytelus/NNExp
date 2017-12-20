import numpy as np
import network_config
import labeled_data

class Network:
    def __init__(self, config : network_config.NetworkConfig):
        np.seterr(all='raise')
        self.config = config

    def feed_forward(self, input_vals):
        for b, w in zip(self.biases, self.weights):
            input_vals = self.activation_c.fn(np.dot(w, input_vals) + b)
        return input_vals

    def sgd(self, labeled_data : labeled_data.LabeledData):
        self.biases = self.config.init_c.get_biases(self.config.neuron_counts)
        self.weights = self.config.init_c.get_weights(self.config.neuron_counts);

        n = len(labeled_data.train)

        mini_batches = [
            labeled_data.train[k : k + self.config.batch_size]
            for k in range(0, n, self.config.batch_size)]

        for j in range(self.config.epochs):
            np.random.shuffle(labeled_data.train)
            for batch in mini_batches:
                self.sgd_batch(batch, len(labeled_data.train))
            print("Epoch %s training complete" % j)

    def sgd_batch(self, batch, n):
        b_batch = [np.zeros(b.shape) for b in self.biases]
        w_batch = [np.zeros(w.shape) for w in self.weights]

        for x, y_true in batch:
            b_backprop, w_backprop = self.backprop(x, y_true)
            b_batch = [bb + bbp for bb, bbp in zip(b_batch, b_backprop)]
            w_batch = [wb + wbp for wb, wbp in zip(w_batch, w_backprop)]

        regularization = 1 - self.config.eta*(self.config.lmbda/n)
        eta_batch = self.config.eta/len(batch)

        self.weights = [regularization * w - eta_batch * wb
                        for w, wb in zip(self.weights, w_batch)]
        self.biases = [b - eta_batch * bb
                       for b, bb in zip(self.biases, b_batch)]

    def backprop(self, x, y_true):
        b_bp = [np.zeros(b.shape) for b in self.biases]
        w_bp = [np.zeros(w.shape) for w in self.weights]

        # feed_forward
        layer_input = x
        layer_inputs = [x] # list to store all the layer_inputs, layer by layer
        input_sums = [] # list to store all the input_sum vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            input_sum = np.dot(w, layer_input) + b
            input_sums.append(input_sum)
            layer_input = self.config.activation_c.fn(input_sum)
            layer_inputs.append(layer_input)

        # backward pass
        d_loss = self.config.loss_c.d_fn(layer_inputs[-1], y_true)
        if d_loss.shape == (10,10):
            print('hit')
        d_activation = self.config.activation_c.d_fn(input_sums[-1])
        delta =  d_loss * d_activation
            
        b_bp[-1] = delta
        w_bp[-1] = np.dot(delta, layer_inputs[-2].transpose())

        for l in range(2, len(self.config.neuron_counts)):
            input_sum = input_sums[-l]
            d_activation = self.config.activation_c.d_fn(input_sum)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * d_activation
            b_bp[-l] = delta
            w_bp[-l] = np.dot(delta, layer_inputs[-l-1].transpose())
        return (b_bp, w_bp)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feed_forward(x)), np.argmax(y_true))
                        for (x, y_true) in test_data]
        return sum(int(y_pred == y_true) for (y_pred, y_true) in test_results)

