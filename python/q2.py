import numpy as np
import scipy.io
from nn import *

"""
The goal of Q2 is to practice parameter tuning

[CODE TO WRITE]
Q 2.1.1     implement train(...) in this file
            train the network and check the validation accuracy
Q 2.1.2     generate a plot
Q 2.1.3     train the network with best_lr, 10*best_lr and 0.1*best_lr
            generate three plots
Q 2.1.4     visualize first layer weights
            compare visualization before and after training
Q 2.1.5     visualize the confusion matrix
Q 2.2.1     train the network with the same structure using Pytorch
Q 2.2.1     train a network
"""


# You don't need to modify from here
#===============================================
# load dataset
train_data = scipy.io.loadmat('/Users/admin/Desktop/ca2-ahn9807/data/train.mat')
valid_data = scipy.io.loadmat('/Users/admin/Desktop/ca2-ahn9807/data/valid.mat')
train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
#===============================================
# to here

# Q 2.1.1 #===============================================
# CODE TO WRITE:
#   implement train(...) in this file
#   train the network and check the validation accuracy
#==== STEP1: Implement train(...)
def train(max_iters = 50, learning_rate = 1e-3, batch_size = 400, validation = False, pretrained = False, pretrained_params = None):
    batches = get_random_batches(train_x, train_y, batch_size)
    valid_batches = get_random_batches(valid_x, valid_y, batch_size)
    batch_num = len(batches)

    # initialize params
    if pretrained:
        params = pretrained_params
    else:
        params={}
        initialize_weights(1024, 64, params, name="layer1")
        initialize_weights(64,36,params, name="output")

    # We'll store loss and acc over iteration
    train_losses, train_acces, valid_losses, valid_acces = [], [], [], []

    for itr in range(max_iters):
        total_loss = 0
        total_acc = 0
        valid_loss = 0
        valid_acc = 0
        for xb, yb in batches:
            # delete "pass" after your implementation
            # CODE TO WIRTE: forward
            h1 = forward(xb, params, name='layer1')
            h2 = forward(h1, params, name='output', activation=softmax)

            # CODE TO WIRTE: loss
            # be sure to add loss and accuracy to epoch totals
            loss, acc = compute_loss_and_acc(yb, h2)
            
            total_loss += loss
            total_acc += acc

            # CODE TO WIRTE: backward
            # be sure that backward(...) in nn.py doesn't include gradient update
            delta = h2-yb
            delta = backwards(delta, params, name='output', activation_deriv=linear_deriv)
            delta = backwards(delta, params, name='layer1', activation_deriv=sigmoid_deriv)

            # CODE TO WIRTE: apply gradient update
            params['W' + 'layer1'] -= learning_rate * params['grad_W' + 'layer1']
            params['b' + 'layer1'] -= learning_rate * params['grad_b' + 'layer1']
            params['W' + 'output'] -= learning_rate * params['grad_W' + 'output']
            params['b' + 'output'] -= learning_rate * params['grad_b' + 'output']

        if validation:
            h1 = forward(valid_x, params, name='layer1')
            h2 = forward(h1, params, name='output', activation=softmax)

            loss, acc = compute_loss_and_acc(valid_y, h2)
        
            valid_losses.append(loss)
            valid_acces.append(acc)

        total_loss = total_loss / batch_num
        total_acc = total_acc / batch_num
        # print loss acc during training
        if itr % 2 == 1:
            print("itr: {:02d} \t loss: {:.2f} \t acc : {:.3f}".format(itr, total_loss, total_acc))

        valid_loss = sum(valid_losses)/len(valid_losses)
        valid_acc = sum(valid_acces)/len(valid_acces)
        train_losses.append(total_loss)
        train_acces.append(total_acc)

    return params, [train_losses, valid_losses], [train_acces, valid_acces]

#==== STEP2: Train the network and check the validation accuracy
max_iters = 3000 #3000
# pick the best batch size, learning rate
batch_size = 6000
learning_rate = 0.00005
hidden_size = 64

# with default settings, you should get training accuracy > 80%
params, result_losses, result_acces = train(2, learning_rate, batch_size, validation=True)
params_10bestlr, result_losses_10bestlr, result_acces_10bestlr = train(2, learning_rate*10, batch_size, validation=True)
params_01bestlr, result_losses_01bestlr, result_acces_01bestlr = train(max_iters, learning_rate*0.1, batch_size, validation=True) 

# should be above 75%
print('Validation accuracy: ', result_acces[1][len(result_losses[1])-1])

"""
# To view the data
for crop in xb:
    import matplotlib.pyplot as plt
    plt.imshow(crop.reshape(32,32).T)
    plt.show()
"""

# Save trained weights
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('Checkpoint is saved!')


# Q 2.1.2 #===============================================
# CODE TO WRITE:
#   generate a plot

import matplotlib.pyplot as plt

fig, (sub1, sub2) = plt.subplots(1, 2)
fig.suptitle("Result of Accuracy and Loss")

sub1.plot(result_losses[0])
sub1.plot(result_losses[1])
sub1.set_ylabel('loss')
sub1.set_xlabel('epoch')
sub1.legend(['train', 'valid'], loc='upper left')

sub2.plot(result_acces[0])
sub2.plot(result_acces[1])
sub2.set_ylabel('accuracy')
sub2.set_xlabel('epoch')
sub2.legend(['train', 'valid'], loc='upper left')
plt.show()

# Q 2.1.3 #===============================================
# CODE TO WRITE:
#   train the network with best_lr, 10*best_lr and 0.1*best_lr
#   generate three plots
fig, (sub1, sub2) = plt.subplots(1, 2)
fig.suptitle("Result of Accuracy and Loss")

sub1.plot(result_losses_10bestlr[0])
sub1.plot(result_losses_10bestlr[1])
sub1.set_ylabel('loss')
sub1.set_xlabel('epoch')
sub1.legend(['train', 'valid'], loc='upper left')

sub2.plot(result_acces_10bestlr[0])
sub2.plot(result_acces_10bestlr[1])
sub2.set_ylabel('accuracy')
sub2.set_xlabel('epoch')
sub2.legend(['train', 'valid'], loc='upper left')
plt.show()

fig, (sub1, sub2) = plt.subplots(1, 2)
fig.suptitle("Result of Accuracy and Loss")

sub1.plot(result_losses_01bestlr[0])
sub1.plot(result_losses_01bestlr[1])
sub1.set_ylabel('loss')
sub1.set_xlabel('epoch')
sub1.legend(['train', 'valid'], loc='upper left')

sub2.plot(result_acces_01bestlr[0])
sub2.plot(result_acces_01bestlr[1])
sub2.set_ylabel('accuracy')
sub2.set_xlabel('epoch')
sub2.legend(['train', 'valid'], loc='upper left')
plt.show()

# Q 2.1.4 #===============================================
# CODE TO WRITE:
#   visualize first layer weights
#   compare visualization before and after training

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

fig = plt.figure(figsize=(8,8))
grid = ImageGrid(fig, 111, nrows_ncols=(8,8))
for ax, w in zip(grid, params['Wlayer1'].T):
    ax.imshow(w.reshape(32,32))
    
plt.show()

# Q 2.1.5 #===============================================
# CODE TO WIRTE:
#   visualize the confusion matrix


# To make us to see the results more clearly, make it in two versions
#   : with and without correct cases
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))

import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()



# Q 2.2.1 #===============================================
# CODE TO WIRTE:
#   train the network with the same structure using Pytorch
import torch



# Q 2.2.2 #===============================================
# CODE TO WIRTE:
#   train a network



