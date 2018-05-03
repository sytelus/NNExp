import experiments
import numpy as np

np.seterr(all='raise')
np.random.seed(42)

# experiments.sq_grad_3layer_300rows()
experiments.nielson_3layer_300rows()
experiments.twin_3layer_300rows()
