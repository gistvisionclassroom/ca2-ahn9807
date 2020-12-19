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
train_data = scipy.io.loadmat('/Users/admin/Desktop/ca2-ahn9807/data/train.mat')
valid_data = scipy.io.loadmat('/Users/admin/Desktop/ca2-ahn9807/data/valid.mat')
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
initialize_weights(1024, 32, params, 'layer1')
initialize_weights(32, 32, params, 'hidden')
initialize_weights(32, 32, params, 'hidden2')
initialize_weights(32, 1024, params, 'output')

def update_parameter(params, p_name, i_name, lr):
    params['m_' + p_name + i_name] = 0.9 * params['m_' + p_name + i_name] - lr * params['grad_' + p_name + i_name]
    params[p_name + i_name] += params['m_' + p_name + i_name]  


losses = []
# should look like your previous training loops
for itr in range(max_iters):
    total_loss = 0
    for xb, _ in batches:
        # training loop can be exactly the same as q1!
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions
        h1 = forward(xb, params, 'layer1', activation=relu)
        h2 = forward(h1, params, 'hidden', activation=relu)
        h3 = forward(h2, params, 'hidden2', activation=relu)
        h4 = forward(h3, params, 'output', activation=sigmoid)

        loss = np.sum((h4-xb) * (h4-xb))
        total_loss += loss

        delta = 2*(h4-xb)
        delta1 = backwards(delta, params, 'output', activation_deriv=sigmoid_deriv)
        delta2 = backwards(delta1, params, 'hidden2', activation_deriv=relu_deriv)
        delta3 = backwards(delta2, params, 'hidden', activation_deriv=relu_deriv)
        delta4 = backwards(delta3, params, 'layer1', activation_deriv=relu_deriv)

        update_parameter(params, 'W', 'layer1', learning_rate)
        update_parameter(params, 'W', 'hidden', learning_rate)
        update_parameter(params, 'W', 'hidden2', learning_rate)
        update_parameter(params, 'W', 'output', learning_rate)
        update_parameter(params, 'b', 'layer1', learning_rate)
        update_parameter(params, 'b', 'hidden', learning_rate)
        update_parameter(params, 'b', 'hidden2', learning_rate)
        update_parameter(params, 'b', 'output', learning_rate)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr, total_loss))
    if itr % lr_rate == lr_rate - 1:
        learning_rate *= 0.9

    losses.append(total_loss)


# Q 3.2.2 #===============================================
# CODE TO WRITE:
#   visualize some (input, output) pairs
import matplotlib
import matplotlib.pyplot as plt

plt.suptitle("Result of Accuracy and Loss")

plt.plot(losses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')

plt.show()

h1 = forward(valid_x, params, 'layer1', relu)
h2 = forward(h1, params, 'hidden', relu)
h3 = forward(h2, params, 'hidden2', relu)
out = forward(h3, params, 'output', sigmoid)
samples = np.random.choice(np.arange(36),5, replace=False)
for i in samples:
    plt.subplot(2, 1, 1)
    plt.imshow(valid_x[i * 100 + 1].reshape(32, 32).T)
    plt.subplot(2, 1, 2)
    plt.imshow(out[i * 100 + 1].reshape(32, 32).T)
    plt.show()
    plt.subplot(2, 1, 1)
    plt.imshow(valid_x[i * 100+2].reshape(32, 32).T)
    plt.subplot(2, 1, 2)
    plt.imshow(out[i * 100+2].reshape(32, 32).T)
    plt.show()

# Q 3.2.3 #===============================================
# CODE TO WRITE:
#   evaluate PSNR
from skimage.measure import compare_psnr as psnr

s_psnr = []
for i in range(valid_x.shape[0]):
    s_psnr.append(psnr(valid_x[i], out[i]))

s_psnr = np.mean(s_psnr)


# should be in range of 13~15
print(s_psnr)