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
train_data = scipy.io.loadmat('../data/train.mat')
valid_data = scipy.io.loadmat('../data/valid.mat')
train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
#===============================================
# to here


# Q 2.1.1 #===============================================
# CODE TO WRITE:
#   implement train(...) in this file
#   train the network and check the validation accuracy


#==== STEP1: Implement train(...)
def train(max_iters = 50, learning_rate = 1e-3, batch_size = 400, validation = False, pretrained = False):
    batches = get_random_batches(train_x, train_y, batch_size)
    batch_num = len(batches)

    # initialize params
    if pretrained:
        params = None
    else:
        params = None

    # We'll store loss and acc over iteration
    train_losses, train_acces, valid_losses, valid_acces = [], [], [], []

    for itr in range(max_iters):
        total_loss = 0
        total_acc = 0
        for xb, yb in batches:
            pass
            # training loop can be exactly the same as q1!


        # validation
        if validation:
            valid_loss, valid_acc = None, None

        # print loss acc during training
        if itr % 2 == 1:
            print("itr: {:02d} \t loss: {:.2f} \t acc : {:.3f}".format(itr, total_loss, total_acc))


#==== STEP2: Train the network and check the validation accuracy
max_iters = 50
# pick the best batch size, learning rate
batch_size = None
learning_rate = None
hidden_size = 64

# with default settings, you should get training accuracy > 80%
params, _, _ = train(max_iters, learning_rate, batch_size)

# should be above 75%
valid_acc = None
print('Validation accuracy: ',valid_acc)

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




# Q 2.1.3 #===============================================
# CODE TO WRITE:
#   train the network with best_lr, 10*best_lr and 0.1*best_lr
#   generate three plots




# Q 2.1.4 #===============================================
# CODE TO WRITE:
#   visualize first layer weights
#   compare visualization before and after training

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid



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



