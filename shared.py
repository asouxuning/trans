import theano
import theano.tensor as T
import numpy as np

x = np.zeros((2,3))
s = theano.shared(value=x, borrow=True)

print(s.eval())

x += 1
print(s.get_value())
