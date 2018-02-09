import theano
import theano.tensor as T
import numpy as np

w1_value = np.random.normal(loc=0.0, scale=0.1, size=(2,2))
b1_value = np.array([0.0,0.0])
w2_value = np.random.normal(loc=0.0, scale=0.1, size=(2,1))
b2_value = np.array([0.0])

theta_value = np.concatenate((w1_value.ravel(), b1_value, w2_value.ravel(), b2_value))
theta = theano.shared(value=theta_value)

w1 = theta[0:4].reshape((2,2))
b1 = theta[4:6].reshape((1,2))
w2 = theta[6:8].reshape((2,1))
b2 = theta[8:9].reshape((1,1))

x = T.matrix('x', dtype=theano.config.floatX)
y = T.matrix('y', dtype=theano.config.floatX)

a0 = x
z1 = T.dot(a0,w1)+b1
a1 = T.nnet.sigmoid(z1)
z2 = T.dot(a1,w2)+b2
a2 = T.nnet.sigmoid(z2)
y_ = a2

cost = T.sum(-y*T.log(y_)-(1-y)*T.log(1-y_))

dtheta = T.grad(cost=cost, wrt=theta)
ddtheta = T.hessian(cost=cost, wrt=theta)

lr = 0.1
updates = [(theta,theta-lr*dtheta)]

# data
X = theano.shared(np.array([[0,0],[0,1],[1,0],[1,1]], dtype=theano.config.floatX))
Y = theano.shared(np.array([[0],[1],[1],[0]], dtype=theano.config.floatX))

index = T.iscalar('index')
givens = [(x,X[index:index+1]), (y,Y[index:index+1])] 
train = theano.function(inputs=[index], outputs=cost, givens=givens, updates=updates)
predict = theano.function(inputs=[x], outputs=y_) 

epoch = 10001
for i in range(epoch):
  for j in range(X.shape.eval()[0]):
    loss = train(j)
    print(loss)
print(predict(X.eval()))

