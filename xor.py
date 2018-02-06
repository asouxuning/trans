import theano
import theano.tensor as T
import numpy as np

w1_initial = np.random.normal(loc=0.0, scale=0.01, size=(2,2)).astype(theano.config.floatX)
w1 = theano.shared(w1_initial)
b1_initial = np.zeros((2,), dtype=theano.config.floatX)
b1 = theano.shared(b1_initial)

w2_initial = np.random.normal(loc=0.0, scale=0.01, size=(2,1)).astype(theano.config.floatX)
w2 = theano.shared(w2_initial)
b2_initial = np.zeros((), dtype=theano.config.floatX)
b2 = theano.shared(b2_initial)

args = [w1,b1,w2,b2]
for i in range(len(args)):
  print(args[i].eval())

x = T.matrix("x", dtype=theano.config.floatX)
y = T.matrix("y", dtype=theano.config.floatX)

z1 = T.dot(x,w1)+b1
a1 = T.nnet.sigmoid(z1)
z2 = T.dot(a1,w2)+b2
a2 = T.nnet.sigmoid(z2)

lambda_ = 0.0
cost = T.sum(-y*T.log(a2)-(1-y)*T.log(1-a2)) \
       + lambda_ * T.sum(w1**2) \
       + lambda_ * T.sum(w2**2)

grads = T.grad(cost=cost, wrt=args)

lr = 0.1
updates = [(arg,arg-lr*grad) for (arg,grad) in zip(args,grads)]

predict = theano.function(inputs=[x], outputs=a2)

train = theano.function(inputs=[x,y], outputs=[cost, a2], updates=updates)

inputs = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=theano.config.floatX)
labels = np.array([[0],[1],[1],[0]], dtype=theano.config.floatX) 

epoch = 10001 
for i in range(epoch):
  for j in range(len(inputs)):
    loss,pred = train(inputs[j:j+1], labels[j:j+1])

print("###")
print(predict(inputs)) 
print("###")
for i in range(len(args)):
  print(args[i].eval())
