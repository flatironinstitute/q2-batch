from pystan import StanModel
import pickle
code = open('batch_nb_single.stan').read()
sm = StanModel(model_code=code)
pickle.dump(sm, open('batch_nb_single.pkl', 'wb'))
