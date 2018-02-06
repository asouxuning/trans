import theano
import theano.tensor as T
import numpy as np

w = T.constant(np.array([1.0,1.0,1.0]))
x = T.vector('x')
b = T.constant(np.array(1.0))

y = T.dot(w,x)+b

func = theano.function(inputs=[], outputs=[y], givens=[(x, np.array([1.0,2.0,3.0]))])

print(func())

