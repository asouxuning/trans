import theano
import theano.tensor as T
import numpy as np

w = T.vector('w', dtype=theano.config.floatX) 
x = T.vector('x', dtype=theano.config.floatX)

y = T.dot(w**3,x**2)

hw,hx = T.hessian(cost=y, wrt=[w,x])

hess = theano.function(inputs=[w,x], outputs=[hw,hx])

x_ = np.array([1.0, 2.0], dtype=theano.config.floatX)
w_ = np.array([1.0, 1.0], dtype=theano.config.floatX)

hw_,hx_ = hess(w_,x_) 

print(hw_)
print(hx_)
