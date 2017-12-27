from keras.optimizers import Optimizer
from keras.legacy import interfaces
from keras import backend as K


class SGDCust(Optimizer):
    """Stochastic gradient descent optimizer.

    # Arguments
        lr: float >= 0. Learning rate.
    """

    def __init__(self, lr=0.01, **kwargs):
        super(SGDCust, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr

        shapes = [K.int_shape(p) for p in params]
        delta_ws = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + delta_ws
        for p, g, delta_wi in zip(params, grads, delta_ws):
            delta_w = - lr * g  # velocity
            self.updates.append(K.update(delta_wi, delta_w))

            new_p = p + delta_w

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr))}
        base_config = super(SGDCust, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
