import numpy as np

class ClassicParamUpdate:
    def fn(self, config, biases, weights, b_batch, w_batch, batch_len, train_len):
        regularization = 1 - config.eta*(config.lmbda/train_len)
        eta_batch = config.eta / batch_len

        weights_new = [regularization * w - eta_batch * wb
                        for w, wb in zip(weights, w_batch)]
        biases_new = [b - eta_batch * bb
                        for b, bb in zip(biases, b_batch)]

        return (biases_new, weights_new)