import theano
import theano.tensor as T
import numpy as np

w1_initial = np.random.normal(loc=0.0, scale=0.1, size=(2,2))
w1 = theano.shared(w1_initial)
b1_initial = np.array([0.0,0.0])
b1 = theano.shared(b1_initial)

w2_initial = np.random.normal(loc=0.0, scale=0.1, size=(2,1))
w2 = theano.shared(w2_initial)
b2_initial = np.array(0.0)
b2 = theano.shared(b2_initial)

args = [w1,b1,w2,b2]

x = T.vector("x", dtype=theano.config.floatX)
y = T.scalar("y", dtype=theano.config.floatX)

z1 = T.dot(x,w1)+b1
a1 = T.nnet.sigmoid(z1)
z2 = T.dot(a1,w2)+b2
a2 = T.nnet.sigmoid(z2)

cost = T.sum(-y*T.log(a2)-(1-y)*T.log(1-a2)) 
grads = T.grad(cost=cost, wrt=args) 

lr = 0.1
updates = [(arg, arg-lr*grad) for (arg,grad) in zip(args,grads)]

predict = theano.function(inputs=[x], outputs=[a2])
train = theano.function(inputs=[x,y], outputs=[a2,cost], updates=updates)

inputs = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=theano.config.floatX)
labels = np.array([0,1,1,0], dtype=theano.config.floatX)

n_epoches = 10001
for i in range(n_epoches):
  for j in range(len(inputs)):
    y_, loss_ = train(inputs[j], labels[j])
    print(loss_)

for i in range(len(inputs)):
  print(predict(inputs[i]))


  
