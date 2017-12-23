import numpy as np

class SigmoidActivation:
    def fn(self, x):
        return 1.0 / (1.0 + np.exp(- x))

    def d_fn(self, x):
        f = self.fn(x)
        return  f * (1 - f)

class ReLUActivation:
    def fn(self, x):
        result = x * (x > 0) # np.maximum(input_sum, 0)
        return result

    def d_fn(self, x):
        result = 1. * (x > 0) # np.greater(x, 0).astype(int)
        return result

class TanhActivation:
    def fn(self, x):
        return numpy.tanh(x)

    def d_fn(self, x):
        f = self.fn(x) 
        return 1. - f * f
