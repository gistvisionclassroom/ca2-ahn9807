import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

# Q 1.1.2
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    W = None # your code here
    b = None # your code here

    temp = np.sqrt(6) / np.sqrt(in_size + out_size)
    W = np.random.uniform(low = -temp, high= temp, size = (in_size, out_size))
    b = np.zeros(out_size)
    
    params['W' + name] = W
    params['b' + name] = b

# Q 1.2.1
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    res = 1/(1+np.exp(-x))
    return res

# Q 1.2.1
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]

    # your code here
    
    pre_act = X.dot(W) + b
    post_act = activation(X.dot(W) + b)

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

# Q 1.2.4
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    sum_exp_x = np.sum(exp_x, axis=1)
    res = np.divide(exp_x, sum_exp_x[:,None])
    
    return res

# Q 1.2.5
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    loss = - (1 / y.shape[0]) * np.sum(y*(np.log(probs)))
    temp = np.zeros_like(probs)
    temp[np.arange(len(probs)), probs.argmax(1)] =1
    acc = (y == temp).all(axis=1).mean()

    return loss, acc

# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

# Q 1.3.1
def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]
    # your code here
    # do the derivative through activation first
    # then compute the derivative W,b, and X

    grad_delta = delta * activation_deriv(post_act)
    grad_W = np.matmul(X.T, grad_delta)
    grad_b = np.sum(grad_delta, axis=0)
    grad_X = np.matmul(grad_delta, W.T)

    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

# Q 1.4.1
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    batches = []
    batch_index = np.arange(x.shape[0])
    np.random.shuffle(batch_index)
    epoch = (int)(x.shape[0] / batch_size)

    new_batch_x = x[batch_index]
    new_batch_y = y[batch_index]
    
    for i in range(epoch):
        batches.append([new_batch_x[i:i+batch_size], new_batch_y[i:i+batch_size]])
        i += batch_size

    return batches
