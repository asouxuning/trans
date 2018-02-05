import theano
import theano.tensor as T
import numpy as np

X = T.matrix("X", dtype=theano.config.floatX)
#y = T.matrix("y", dtype=theano.config.floatX)
#y = T.vector("y", dtype=theano.config.floatX)
y = T.scalar("y", dtype=theano.config.floatX)
#mtype = T.TensorType(dtype=theano.config.floatX, broadcastable=(True, False))
#mtype = T.TensorType(dtype=theano.config.floatX, broadcastable=(False,))
#y = mtype("y")
z = X + y

f = theano.function(inputs=[X,y], outputs=[z])

#print(f(np.ones((3,2), dtype=theano.config.floatX), np.ones((1,2), dtype=theano.config.floatX)))
#print(f(np.ones((3,2), dtype=theano.config.floatX), np.ones((2,), dtype=theano.config.floatX)))
print(f(np.ones((3,2), dtype=theano.config.floatX), np.ones((), dtype=theano.config.floatX)))

print(X.broadcastable)
print(y.broadcastable)
