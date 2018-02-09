import theano
import theano.tensor as T
import numpy as np

w1_value = np.random.normal(loc=0.0, scale=0.1, size=(2,2))
b1_value = np.array([0.0])
w2_value = np.random.normal(loc=0.0, scale=0.1, size=(2,1))
b2_value = np.array([0.0])

theta_value = np.concatenate((w1_value.ravel(), b1_value, w2_value.ravel(), b2_value))
theta = theano.shared(value=theta_value)

print(theta.eval())
