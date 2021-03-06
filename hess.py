import theano
import theano.tensor as T
import numpy as np

<<<<<<< HEAD
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
=======
t = T.vector('t')
x0 = t[0]
x1 = t[1]
x2 = t[2]

y = 3*x0**3 + 2*x1**2 + x0*x1 + x1*x2 + x2

dx0,dx1,dx2= T.grad(cost=y, wrt=[x0,x1,x2])
J = T.grad(cost=y, wrt=t)
H = T.hessian(cost=y, wrt=t)

feed_dict = {x0:1,x1:1,x2:1}
print(dx0.eval(feed_dict))
print(dx1.eval(feed_dict))
print(dx2.eval(feed_dict))

feed_dict = {t:np.array([1,1,1])}
print(J.eval(feed_dict))
print(H.eval(feed_dict))

>>>>>>> 134caca0afe214f9d8f8182977eedcdd111e5873
