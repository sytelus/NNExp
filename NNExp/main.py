import experiments.experiments as exp
import numpy as np

# np.seterr(all='raise')
np.random.seed(42)

exp.nielson_3layer_300rows()

#exp.twin_3layer_300rows()
#exp.delta_boosted_3layer_300rows()
#exp.neilson_5layer_200rows_100epochs()
#exp.delta_boosted_5layer_300rows()
##exp.sq_grad_3layer_300rows()