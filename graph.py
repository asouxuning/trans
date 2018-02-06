import theano
import theano.tensor as T
import numpy as np

x = T.matrix('x')
y = T.vector('y')
c = T.constant(np.array(1))

z = x + y + c
w = x * y + c 

func = theano.function(inputs=[x,y], outputs=[z,w])

print(func(np.array([[1,2],[2,3],[3,4]]), np.ones((2,))))

