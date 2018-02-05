import theano
import theano.tensor as T
import numpy as np
import sys

w1_initial = np.random.normal(loc=0.0, scale=0.01, size=(2,2)).astype(theano.config.floatX)
w1 = theano.shared(w1_initial)
#b1_initial = np.zeros((1,2), dtype=theano.config.floatX)
b1_initial = np.zeros((2,), dtype=theano.config.floatX)
b1 = theano.shared(b1_initial)

w2_initial = np.random.normal(loc=0.0, scale=0.01, size=(2,1)).astype(theano.config.floatX)
w2 = theano.shared(w2_initial)
#b2_initial = np.zeros((1,1), dtype=theano.config.floatX)
b2_initial = np.zeros((), dtype=theano.config.floatX)
b2 = theano.shared(b2_initial)

args = [w1,b1,w2,b2]

x = T.matrix("x", dtype=theano.config.floatX)
y = T.matrix("y", dtype=theano.config.floatX)

z0 = x
z1 = T.dot(z0,w1)+b1
a1 = T.nnet.sigmoid(z1)
z2 = T.dot(z1,w2)+b2
a2 = T.nnet.sigmoid(z2)
y_ = a2

cost = T.sum(-y*T.log(y_)-(1-y)*T.log(1-y_))
# cost = T.sum((y-y_)**2)

grads = T.grad(cost=cost, wrt=args)

lr = 0.01
updates = [(arg,arg-lr*grad) for (arg,grad) in zip(args,grads)]

predict = theano.function(inputs=[x], outputs=y_)

train = theano.function(inputs=[x,y], outputs=[cost, y_], updates=updates)

inputs = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=theano.config.floatX)
outputs = np.array([[0],[1],[1],[0]], dtype=theano.config.floatX) 

epoch = 100001 
for i in range(epoch):
  for j in range(len(inputs)):
    loss,pred = train(inputs[j:j+1], outputs[j:j+1])
    print(loss)

print(predict(inputs))



