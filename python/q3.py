import numpy as np
import scipy.io
from nn import *
from collections import Counter

"""
Q3

[CODE TO WRITE]
Q 3.1.1     implement the training loop following the given instruction
Q 3.1.2     implement momentum
Q 3.2.1     train with the default settings
Q 3.2.2     visualize some (input, output) pairs
Q 3.2.3     evaluate PSNR
"""


# You don't need to modify from here
#===============================================
train_data = scipy.io.loadmat('../data/train.mat')
valid_data = scipy.io.loadmat('../data/valid.mat')
# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']
#===============================================
# to here


# Q 3.1.1 #===============================================
# Q 3.1.2 #===============================================
# Q 3.2.1 #===============================================
# CODE TO WRITE:
#   implement the training loop following the given instruction
#   implement momentum
#   train with the default settings

max_iters = 100
# pick a batch size, learning rate
batch_size = 36
learning_rate = 3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x, np.ones((train_x.shape[0], 1)), batch_size)
batch_num = len(batches)

# Hint : search for Counter() then you'll find that it's super-useful in this section.
params = Counter()

# initialize layers here


# should look like your previous training loops
for itr in range(max_iters):
    total_loss = 0
    for xb, _ in batches:
        pass
        # training loop can be exactly the same as q1!
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr, total_loss))
    if itr % lr_rate == lr_rate - 1:
        learning_rate *= 0.9




# Q 3.2.2 #===============================================
# CODE TO WRITE:
#   visualize some (input, output) pairs
import matplotlib
import matplotlib.pyplot as plt

h1 = forward(xb, params, 'layer1', relu)
h2 = forward(h1, params, 'hidden', relu)
h3 = forward(h2, params, 'hidden2', relu)
out = forward(h3, params, 'output', sigmoid)
for i in range(5):
    plt.subplot(2, 1, 1)
    plt.imshow(xb[i].reshape(32, 32).T)
    plt.subplot(2, 1, 2)
    plt.imshow(out[i].reshape(32, 32).T)
    plt.show()




# Q 3.2.3 #===============================================
# CODE TO WRITE:
#   evaluate PSNR
from skimage.measure import compare_psnr as psnr

psnr = None

# should be in range of 13~15
print(psnr)