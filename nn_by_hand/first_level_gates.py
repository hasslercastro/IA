import numpy as np

def sigmoid(z):
    return 1 / (1 + (np.exp(-z)))

def sigmoid_prime(z):
    return sigmoid(z) * ( 1 - sigmoid(z))

def forward(X, w, b):
    z = sigmoid((w @ X) + b)
    return z

def cost(y , y_hat):
    return y - y_hat

def back_and_grad(X , y , w , b, learning_rate , num_ep):

    for i in range(num_ep):
        z = forward(X,w,b)
        error = cost( y , z)
        der_w = (error * sigmoid_prime(z)) @  X.T
        der_b = error @ sigmoid_prime(z).T
        w = w + learning_rate * der_w
        b = b + learning_rate * der_b
        print(z)

    

w = np.random.rand(1,2)
b = np.random.rand(1)
X = np.array([[1,1,0,0], [1,0,1,0]])
y = np.array([1,1,1,0])
learning_rate = 0.001
back_and_grad(X , y , w, b , learning_rate, 20000)
