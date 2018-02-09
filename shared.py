import theano
import theano.tensor as T
import numpy as np

<<<<<<< HEAD
x = np.zeros((2,3))
s = theano.shared(value=x, borrow=True)

print(s.eval())

x += 1
print(s.get_value())
=======
x = theano.shared(np.zeros((2,2)))

acc = theano.function(inputs=[], outputs=[x], updates=[(x,x+1)])
print(acc())
print(acc())

>>>>>>> bef74cd785177eebde33cb212f8b3872c4456704
