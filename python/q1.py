import numpy as np
# you should write your functions in nn.py
from nn import *
from util import *

"""
The goal of Q1 is to understand the structure of neural network and 
implement it by oneself, without using any already-made ML libraries.

Most of modification will be in nn.py. Since you have to reuse all the functions in nn.py, 
make sure you implement them correctly.

[CODE TO WRITE]
Q 1.1.2.    initialize_weights(in_size,out_size,params,name='') in nn.py
Q 1.2.1.    sigmoid(x), forward(X,params,name='',activation=sigmoid) in nn.py
Q 1.2.4.    softmax(x) in nn.py
Q 1.2.5.    compute_loss_and_acc(y, probs) in nn.py
Q 1.3.1.    backwards(delta,params,name='',activation_deriv=sigmoid_deriv) in nn.py
Q 1.4.1.    get_random_batches(x,y,batch_size) in nn.py
            training loop in this file
Q 1.5.1.    forward & backward pass by computing numerical/analytical gradients in this file
"""


# You don't need to modify from here
#===============================================
# fake data
# feel free to plot it in 2D
# what do you think these 4 classes are?
g0 = np.random.multivariate_normal([3.6,40],[[0.05,0],[0,10]],10)
g1 = np.random.multivariate_normal([3.9,10],[[0.01,0],[0,5]],10)
g2 = np.random.multivariate_normal([3.4,30],[[0.25,0],[0,5]],10)
g3 = np.random.multivariate_normal([2.0,10],[[0.5,0],[0,10]],10)
x = np.vstack([g0,g1,g2,g3])
# we will do XW + B
# This implies that the data is N x D

# create labels
y_idx = np.array([0 for _ in range(10)] + [1 for _ in range(10)] + [2 for _ in range(10)] + [3 for _ in range(10)])
# convert to one_hot
y = np.zeros((y_idx.shape[0],y_idx.max()+1))
y[np.arange(y_idx.shape[0]),y_idx] = 1

# parameters in a dictionary
params = {}
#=================================================
# to here


# Q 1.1.2 #=================================================
# initialize a layer
# CODE TO WRITE: initialize_weights(in_size,out_size,params,name='') in nn.py
initialize_weights(2,25,params,'layer1')
initialize_weights(25,4,params,'output')
assert(params['Wlayer1'].shape == (2,25))
assert(params['blayer1'].shape == (25,))
# Check if it prints: 0, [0.05 to 0.12]
print("{}, {:.2f}".format(params['blayer1'].sum(),params['Wlayer1'].std()**2))
print("{}, {:.2f}".format(params['boutput'].sum(),params['Woutput'].std()**2))




# Q 1.2.1 #=================================================
# implement sigmoid
# CODE TO WRITE: sigmoid(x) in nn.py
test = sigmoid(np.array([-1000,1000]))
# Check if it prints 0 and 1:
print(test.min(),test.max())

# implement forward
# CODE TO WRITE: forward(X,params,name='',activation=sigmoid) in nn.py
h1 = forward(x,params,'layer1')
# Check if the shape of h1 is correct
print(h1.shape)




# Q 1.2.4 #=================================================
# implement softmax
# CODE TO WIRTE: softmax(x) in nn.py
probs = forward(h1,params,'output',softmax)
# Check if it prints some positive value, ~1, ~1, (40,4)
# make sure you understand these values!
print(probs.min(),min(probs.sum(1)),max(probs.sum(1)),probs.shape)




# Q 1.2.5 #=================================================
# implement compute_loss_and_acc using cross-entropy loss
# CODE TO WRITE: compute_loss_and_acc(y, probs) in nn.py
loss, acc = compute_loss_and_acc(y, probs)
# should be around -np.log(0.25)*40 [~55] and 0.25
# if it is not, check softmax!
print("{}, {:.2f}".format(loss,acc))




# Q 1.3.1 #=================================================
# implement backpropagation
# CODE TO WRITE: backwards(delta,params,name='',activation_deriv=sigmoid_deriv) in  nn.py

# To make it easier, let the derivative of cross-entropy(softmax(x)) is probs - 1[correct actions]
delta1 = probs
delta1[np.arange(probs.shape[0]),y_idx] -= 1
# We already got the simplified derivatives from softmax
# So, instead of writing exceptional lines, we pass in a linear_deriv, which is just a vector of ones.
delta2 = backwards(delta1,params,'output',linear_deriv)
backwards(delta2,params,'layer1',sigmoid_deriv)

# Check if W and b match their gradients sizes
for k,v in sorted(list(params.items())):
    if 'grad' in k:
        name = k.split('_')[1]
        print(name,v.shape, params[name].shape)




# Q 1.4.1 #=================================================
# implement random batch division
# CODE TO WRITE: get_random_batches(x,y,batch_size) in nn.py
batches = get_random_batches(x,y,5)
# print batch sizes
print([_[0].shape[0] for _ in batches])
batch_num = len(batches)

# implement training loop in this file
# CODE TO WRITE: forward, loss, backward, gradient update
max_iters = 500
learning_rate = 1e-3
# with default settings, you should get loss < 35 and accuracy > 75%
for itr in range(max_iters):
    total_loss = 0
    avg_acc = 0
    for xb,yb in batches:
        pass # delete "pass" after your implementation

        # CODE TO WIRTE: forward


        # CODE TO WIRTE: loss
        # be sure to add loss and accuracy to epoch totals


        # CODE TO WIRTE: backward
        # be sure that backward(...) in nn.py doesn't include gradient update


        # CODE TO WIRTE: apply gradient update


        
    if itr % 100 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,avg_acc))




# Q 1.5.1 #=================================================
# The goal is to check gaps between numerical gradients and analytical gradients.
# you can do this before or after training the network.
# CODE TO WRITE: forward & backward pass by computing numerical/analytical gradients in this file (but do not apply it)

# Before start, save the old params, including gradeints you've just computed
import copy
# Make sure to make copy using deepcopy!
params_orig = copy.deepcopy(params)


#==== STEP1: Get numerical gradients
# we'll try to get the same results with numerical gradients

params_eps = {} # new params using numerical gradients
eps = 1e-6
for k,v in params.items():
    if '_' in k:
        continue

    # we have a real parameter!
    grad_v = np.zeros_like(v)

    # CODE TO WRITE:
    # for each value inside the parameter
    #   add/substract epsilon
    #   run the network
    #   get the loss
    #   compute derivatives with central differences
    #   store them inside the params_eps

    # store numerical gradients in params_eps
    params_eps['grad_' + k] = grad_v


#==== STEP2: Get analytical gradients
# Hint: You've done it in the earlier question
# CODE TO WRITE: update params_orig once



#==== STEP3 : Check gaps between numerical gradients and analytical gradients
total_error = 0
for k in params.keys():
    if 'grad_' in k:
        # relative error
        err = np.abs(params_eps[k] - params_orig[k])/np.maximum(np.abs(params_eps[k]),np.abs(params_orig[k]))
        err = err.sum()
        print('{} {:.2e}'.format(k, err))
        total_error += err
# Check if it is less than 1e-4. (It should be.)
print('total {:.2e}'.format(total_error))
