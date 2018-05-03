import experiments
import numpy as np

# np.seterr(all='raise')
np.random.seed(42)

# experiments.sq_grad_3layer_300rows()
# experiments.nielson_3layer_300rows()
#experiments.twin_3layer_300rows()

#experiments.delta_boosted_3layer_300rows()
#experiments.neilson_5layer_200rows_100epochs()
experiments.delta_unboosted_5layer_300rows()
