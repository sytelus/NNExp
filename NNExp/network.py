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

    def _init_nn_params(self, other = None):
        if other is None:
            self.biases = self.config.init_c.get_biases(self.config.neuron_counts)
            self.weights = self.config.init_c.get_weights(self.config.neuron_counts)
        else:
            self.biases = other.biases
            self.weights = other.weights

    def train(self, train_data, validate_date):
        self._init_nn_params()

        n = len(train_data)

        # run epochs
        for j in range(self.config.epochs):
            # for each epoch, form mini-batches
            np.random.shuffle(train_data)
            mini_batches = [
                train_data[k : k + self.config.batch_size]
                for k in range(0, n, self.config.batch_size)]
            for batch in mini_batches:
                total_loss, db_sum, dw_sum = self._train_batch(batch, n)

                # make update to weights and biases for entire batch
                self._update_nn_params(total_loss, db_sum, dw_sum, len(batch), n)
            # output results
            print("Epoch %d : %d / %d" % (j, self.test(validate_date), len(validate_date)))

        # end of training
        print('done')

    def _update_nn_params(self, total_loss, b_batch, w_batch, batch_len, train_len):
        self.biases, self.weights = self.config.param_update_c.fn(
            self.config, self.biases, self.weights, 
            total_loss, b_batch, w_batch, batch_len, train_len)


    def _train_batch(self, batch, n):
        # init sums with zeros
        dw_sum = [np.zeros(w.shape) for w in self.weights]
        db_sum = [np.zeros(b.shape) for b in self.biases]

        total_loss = 0
        # for data point in batch
        for x, y_true, id in batch:
            # do the backprop to find dW and dB
            db_batch, dw_batch, loss = self.config.backprop_c.fn(
                self.config, self.biases, self.weights, x, y_true)

            # accumulate result of the backprop on each batch
            total_loss += loss
            dw_sum = [wb + wbp for wb, wbp in zip(dw_sum, dw_batch)]
            db_sum = [bb + bbp for bb, bbp in zip(db_sum, db_batch)]

        return (total_loss, db_sum, dw_sum)

    def test(self, test_data):
        test_results = [(np.argmax(self._feed_forward(x)), np.argmax(y_true))
                        for (x, y_true, id) in test_data]
        return sum(int(y_pred == y_true) for (y_pred, y_true) in test_results)

