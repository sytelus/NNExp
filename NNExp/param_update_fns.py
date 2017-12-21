import numpy as np
import math

class ClassicParamUpdate:
    def fn(self, config, biases, weights, total_loss,
        b_batch, w_batch, batch_len, train_len):

        regularization = 1 - config.eta*(config.lmbda/train_len)
        eta_batch = config.eta / batch_len

        weights_new = [regularization * w - eta_batch * wb
                        for w, wb in zip(weights, w_batch)]
        biases_new = [b - eta_batch * bb
                        for b, bb in zip(biases, b_batch)]

        return (biases_new, weights_new)


class GdFixedParamUpdate:
    def fn(self, config, biases, weights, total_loss,
        b_batch, w_batch, batch_len, train_len):

        regularization = 1 - config.eta*(config.lmbda/train_len)
        eta_batch = config.eta / batch_len

        weights_new = [regularization * w - GdFixedParamUpdate.fixed_delta(total_loss,wb, regularization * w)
                        for w, wb in zip(weights, w_batch)]
        biases_new = [b - GdFixedParamUpdate.fixed_delta(total_loss, bb, b)
                        for b, bb in zip(biases, b_batch)]

        return (biases_new, weights_new)

    @staticmethod
    def fixed_delta(total_loss, delta, max_val):
        total_loss = math.fabs(total_loss) / 3
        #max_val = np.fabs(max_val)
        #max_val = 0.1 * np.where(max_val < 1E-100, 1E-100, max_val)

        delta = np.where(delta == 0, 1E-30, delta)
        fixed = np.clip(total_loss/delta, -total_loss, total_loss)
        return fixed