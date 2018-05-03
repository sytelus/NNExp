import numpy as np
import network_config
import labeled_data

class Network:
    def __init__(self, config : network_config.NetworkConfig):
        self.config = config

    def _feed_forward(self, input_vals):
        for b, w in zip(self.biases, self.weights):
            input_vals = self.config.activation_c.fn(np.dot(w, input_vals) + b)
        return input_vals

    def train(self, train_data, validate_date):
        # intialize weights and biases
        self.biases = self.config.init_c.get_biases(self.config.neuron_counts)
        self.weights = self.config.init_c.get_weights(self.config.neuron_counts);

        n = len(train_data)

        # run epochs
        for j in range(self.config.epochs):
            # for each epoch, form mini-batches
            np.random.shuffle(train_data)
            mini_batches = [
                train_data[k : k + self.config.batch_size]
                for k in range(0, n, self.config.batch_size)]
            for batch in mini_batches:
                self._train_batch(batch, len(train_data))

            # output results
            print("Epoch %d : %d / %d" % (j, self.test(validate_date), len(validate_date)))

        # end of training
        print('done')


    def _train_batch(self, batch, n):
        # init sums with zeros
        dw_sum = [np.zeros(b.shape) for b in self.biases]
        db_sum = [np.zeros(w.shape) for w in self.weights]

        total_loss = 0
        # for data point in batch
        for x, y_true, id in batch:
            # do the backprop to find dW and dB
            db_batch, dw_batch, loss = self.config.backprop_c.fn(
                self.config, self.biases, self.weights, x, y_true)

            # accumulate result of the backprop on each batch
            total_loss += loss
            dw_sum = [bb + bbp for bb, bbp in zip(dw_sum, db_batch)]
            db_sum = [wb + wbp for wb, wbp in zip(db_sum, dw_batch)]

        # make update to weights and biases for entire batch
        self.biases, self.weights = self.config.param_update_c.fn(self.config, self.biases, self.weights, 
            total_loss, dw_sum, db_sum, len(batch), n)


    def test(self, test_data):
        test_results = [(np.argmax(self._feed_forward(x)), np.argmax(y_true))
                        for (x, y_true, id) in test_data]
        return sum(int(y_pred == y_true) for (y_pred, y_true) in test_results)

