import theano
import theano.tensor as T
import numpy as np

x = theano.shared(np.zeros((2,2)))

acc = theano.function(inputs=[], outputs=[x], updates=[(x,x+1)])
print(acc())
print(acc())

