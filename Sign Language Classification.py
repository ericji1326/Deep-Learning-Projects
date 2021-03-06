# -*- coding: utf-8 -*-
"""Copy of Lab 3 - Eric Ji (1005108358).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-Ce3MQqKya1zOPz9CrtHETVQUCfnwyCV
"""

#mount googledrive
from google.colab import drive
drive.mount('/content/gdrive')

!pip install split-folders

import splitfolders
input_path = '/content/gdrive/My Drive/APS360/Lab 3/Gesture_Dataset'
output_path = '/content/gdrive/My Drive/APS360/Lab 3/Split Data'
#splitting the data 60/20/20
splitfolders.ratio(input_path, output=output_path, seed=1, ratio=(.6, .2, .2))

# Loading these images from Drive

import torch
import numpy as np

import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch.utils.data.sampler import SubsetRandomSampler

# location on Google Drive
master_path = '/content/gdrive/My Drive/APS360/Lab 3/Split Data/' 

# Transform Settings - Do not use RandomResizedCrop
transform = transforms.Compose([transforms.Resize((224,224)), 
                                transforms.ToTensor()])

# As performed by code above, there are 3 folders with 60% training, 20% validation and 20% testing samples
train_dataset = torchvision.datasets.ImageFolder(master_path + 'train', transform=transform)
val_dataset = torchvision.datasets.ImageFolder(master_path + 'val', transform=transform)
test_dataset = torchvision.datasets.ImageFolder(master_path + 'test', transform=transform)

# Prepare Dataloader
batch_size = 27
num_workers = 1

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)


# Visualize some sample data
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']

# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()           # convert images to numpy for display

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    plt.imshow(np.transpose(images[idx], (1, 2, 0)))
    ax.set_title(classes[labels[idx]])

# Model Architecture 

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim                                                     # for gradient descent
import matplotlib.pyplot as plt
import numpy as np

# Creating a CNN
class CNNClassifier(nn.Module):

    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.name = "net"                                                       
        self.conv1 = nn.Conv2d(3,5,5)                                           # First kernel is a 5 by 5, 3 color channels, it has output of 5    -> given 224x224x3, you are left with 220x220x5
        self.pool = nn.MaxPool2d(2,2)                                           # Max pooling layer with kernel size 2 and stride 2                 -> you are left with 110x110x5
        self.conv2 = nn.Conv2d(5,16,5)                                          # Second kernel is 5 by 5, it changes input depth from 5 to 16      -> you are left with 106x106x16
                                                                                # After second Max pooling layer                                    -> you are left with 53x53x16
        self.fc1 = nn.Linear(53*53*16, 100)                                     # ANN portion
        self.fc2 = nn.Linear(100, 64)
        self.fc3 = nn.Linear(64,9)                                              # 9 possible outputs

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))                                    # Apply first kernel, then activation function, then max pooling 
        x = self.pool(F.relu(self.conv2(x)))                                    # Apply second kernel, then activation function, then max pooling 
        x = x.view(-1, 53*53*16)                                                # flatten tensor for ANN portion
        x = F.relu(self.fc1(x))                                                 # Apply activation function on first fully connected layer
        x = F.relu(self.fc2(x))                                                 # Apply activation function on second fully connected layer
        x = self.fc3(x)                                                         # final activation function is included with criterion
        return x

# Training Code

import time

def train(model, train_loader, val_loader, batch_size=27, num_epochs=1, learn_rate = 0.001):

    torch.manual_seed(1000)                                                     # Set manual seed
    criterion = nn.CrossEntropyLoss()                                           # Set criterion
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)                   # Set optimizer as Adam Optimizer 

    train_acc, val_acc = [], []

    # training
    start_time = time.time()
    print ("Training Started...")
    n = 0 # the number of iterations
    for epoch in range(num_epochs):
        for imgs, labels in iter(train_loader):
            
            if use_cuda and torch.cuda.is_available():
              imgs = imgs.cuda()
              labels = labels.cuda()

            out = model(imgs)                                                   # forward pass
            loss = criterion(out, labels)                                       # compute the total loss
            loss.backward()                                                     # backward pass (compute parameter updates)
            optimizer.step()                                                    # make the updates for each parameter
            optimizer.zero_grad()                                               # a clean up step for PyTorch
            n += 1
            
        # track accuracy
        train_acc.append(get_accuracy(model, train_loader))
        val_acc.append(get_accuracy(model, val_loader))
        print(epoch, train_acc[-1], val_acc[-1])

    print('Finished Training')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Total time elapsed: {:.2f} seconds".format(elapsed_time))
            
    return train_acc, val_acc

# Calculate Accuracy

def test_accuracy(model, data_loader):
    correct = 0
    total = 0
    for imgs, labels in data_loader:
        
        if use_cuda and torch.cuda.is_available():
          imgs = imgs.cuda()
          labels = labels.cuda()

        output = model(imgs)
        #select index with maximum prediction score
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.shape[0]
    return correct / total

# Splitting the data like before in a 80/20 split
import splitfolders
path = '/content/gdrive/My Drive/APS360/Lab 3/Personal_Dataset'
output_path = '/content/gdrive/My Drive/APS360/Lab 3/Split_Personal_Data'

splitfolders.ratio(path, output=output_path, ratio=(.8, .2))

split_path = '/content/gdrive/My Drive/APS360/Lab 3/Split_Personal_Data/'

train_overfit = torchvision.datasets.ImageFolder(split_path + 'train', transform=transform)
val_overfit = torchvision.datasets.ImageFolder(split_path + 'val', transform=transform)

# Overfit Test (Sanity Check)

train_loader_personal = torch.utils.data.DataLoader(train_overfit, batch_size=batch_size, 
                                           num_workers=num_workers, shuffle=True)
val_loader_personal = torch.utils.data.DataLoader(val_overfit, batch_size=batch_size, 
                                           num_workers=num_workers, shuffle=True)

overfit_tester = CNNClassifier()

use_cuda = True                                                                 # Attempt to use Cuda

if torch.cuda.is_available():                                                   # Will return true if settings configured correctly
    print("Using Cuda")
    overfit_tester.cuda()

train_acc, val_acc = train(overfit_tester, train_loader_personal, val_loader_personal, batch_size=27, num_epochs=30, learn_rate = 0.001)

import matplotlib.pyplot as plt

plt.plot(range(0,30), train_acc, label='Train')                                 # plotting the training accuracy
plt.plot(range(0,30), val_acc, label='Validation')                              # plotting the validation accuracy
plt.legend(loc='lower right')

# Trial 1

orig_test = CNNClassifier()

use_cuda = True                                                                 # Attempt to use Cuda

if torch.cuda.is_available():                                                   # Will return true if settings configured correctly
    print("Using Cuda")
    orig_test.cuda()

train_acc, val_acc = train(orig_test, train_loader, val_loader, batch_size=27, num_epochs=30, learn_rate = 0.001)

plt.plot(range(0,30), train_acc, label='Train')                                 # plotting the training accuracy
plt.plot(range(0,30), val_acc, label='Validation')                              # plotting the validation accuracy
plt.legend(loc='lower right')

# Trial 2
# changing batchsize from 27 to 54

batchsize_test = CNNClassifier()

use_cuda = True                                                                 # Attempt to use Cuda

if torch.cuda.is_available():                                                   # Will return true if settings configured correctly
    print("Using Cuda")
    batchsize_test.cuda()

train_acc, val_acc = train(batchsize_test, train_loader, val_loader, batch_size=54, num_epochs=30, learn_rate = 0.001)

plt.plot(range(0,30), train_acc, label='Train')                                 # plotting the training accuracy
plt.plot(range(0,30), val_acc, label='Validation')                              # plotting the validation accuracy
plt.legend(loc='lower right')

# Trial 3
# changing learning rate from 0.001 to 0.01

lr_test = CNNClassifier()

use_cuda = True                                                                 # Attempt to use Cuda

if torch.cuda.is_available():                                                   # Will return true if settings configured correctly
    print("Using Cuda")
    lr_test.cuda()

train_acc, val_acc = train(lr_test, train_loader, val_loader, batch_size=27, num_epochs=30, learn_rate = 0.01)

plt.plot(range(0,30), train_acc, label='Train')                                 # plotting the training accuracy
plt.plot(range(0,30), val_acc, label='Validation')                              # plotting the validation accuracy
plt.legend(loc='lower right')

# Trial 4
# changing learning rate to 0.001
# changing batchsize to 64
# epochs to 40

random_test = CNNClassifier()

use_cuda = True                                                                 # Attempt to use Cuda

if torch.cuda.is_available():                                                   # Will return true if settings configured correctly
    print("Using Cuda")
    random_test.cuda()

train_acc, val_acc = train(random_test, train_loader, val_loader, batch_size=64, num_epochs=40, learn_rate = 0.001)

plt.plot(range(0,40), train_acc, label='Train')                                 # plotting the training accuracy
plt.plot(range(0,40), val_acc, label='Validation')                              # plotting the validation accuracy
plt.legend(loc='lower right')

# Trial 5
# Increasing the number of fully connected layers

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim                     #for gradient descent
import matplotlib.pyplot as plt
import numpy as np

# Creating a CNN
class CNNClassifier1(nn.Module):

    def __init__(self):
        super(CNNClassifier1, self).__init__()
        self.name = "net"                                                       
        self.conv1 = nn.Conv2d(3,5,5)                                           # First kernel is a 5 by 5, 3 color channels, it has output of 5    -> given 224x224x3, you are left with 220x220x5
        self.pool = nn.MaxPool2d(2,2)                                           # Max pooling layer with kernel size 2 and stride 2                 -> you are left with 110x110x5
        self.conv2 = nn.Conv2d(5,16,5)                                          # Second kernel is 5 by 5, it changes input depth from 5 to 16      -> you are left with 106x106x16
                                                                                # After second Max pooling layer                                    -> you are left with 53x53x16
        self.fc1 = nn.Linear(53*53*16, 100)                                     # ANN portion
        self.fc2 = nn.Linear(100, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32,9)                                              # 9 possible outputs

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))                                    # Apply first kernel, then activation function, then max pooling 
        x = self.pool(F.relu(self.conv2(x)))                                    # Apply second kernel, then activation function, then max pooling 
        x = x.view(-1, 53*53*16)                                                # flatten tensor for ANN portion
        x = F.relu(self.fc1(x))                                                 # Apply activation function on first fully connected layer
        x = F.relu(self.fc2(x))                                                 # Apply activation function on second fully connected layer
        x = F.relu(self.fc3(x)) 
        x = self.fc4(x)                                                         # final activation function is included with criterion
        return x



layer_test = CNNClassifier1()

use_cuda = True                                                                 # Attempt to use Cuda

if torch.cuda.is_available():                                                   # Will return true if settings configured correctly
    print("Using Cuda")
    layer_test.cuda()

train_acc, val_acc = train(layer_test, train_loader, val_loader, batch_size=54, num_epochs=30, learn_rate = 0.001)

plt.plot(range(0,30), train_acc, label='Train')                                 # plotting the training accuracy
plt.plot(range(0,30), val_acc, label='Validation')                              # plotting the validation accuracy
plt.legend(loc='lower right')

# Trial 6
# Decreasing the number of fully connected layers

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim                     #for gradient descent
import matplotlib.pyplot as plt
import numpy as np

# Creating a CNN
class CNNClassifier2(nn.Module):

    def __init__(self):
        super(CNNClassifier2, self).__init__()
        self.name = "net"                                                       
        self.conv1 = nn.Conv2d(3,5,5)                                           # First kernel is a 5 by 5, 3 color channels, it has output of 5    -> given 224x224x3, you are left with 220x220x5
        self.pool = nn.MaxPool2d(2,2)                                           # Max pooling layer with kernel size 2 and stride 2                 -> you are left with 110x110x5
        self.conv2 = nn.Conv2d(5,16,5)                                          # Second kernel is 5 by 5, it changes input depth from 5 to 16      -> you are left with 106x106x16
                                                                                # After second Max pooling layer                                    -> you are left with 53x53x16
        self.fc1 = nn.Linear(53*53*16, 32)                                     # ANN portion
        self.fc2 = nn.Linear(32, 9)
       
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))                                    # Apply first kernel, then activation function, then max pooling 
        x = self.pool(F.relu(self.conv2(x)))                                    # Apply second kernel, then activation function, then max pooling 
        x = x.view(-1, 53*53*16)                                                # flatten tensor for ANN portion
        x = F.relu(self.fc1(x))                                                 # Apply activation function on first fully connected layer
        x = self.fc2(x)                                               
                                                     
        return x



layer_test = CNNClassifier2()

use_cuda = True                                                                 # Attempt to use Cuda

if torch.cuda.is_available():                                                   # Will return true if settings configured correctly
    print("Using Cuda")
    layer_test.cuda()

train_acc, val_acc = train(layer_test, train_loader, val_loader, batch_size=54, num_epochs=30, learn_rate = 0.001)

plt.plot(range(0,30), train_acc, label='Train')                                 # plotting the training accuracy
plt.plot(range(0,30), val_acc, label='Validation')                              # plotting the validation accuracy
plt.legend(loc='lower right')

# Test on Best Model

test_acc = get_accuracy(batchsize_test, test_loader)
print(test_acc)

# Apply Transfer Learning

import torchvision.models
alexnet = torchvision.models.alexnet(pretrained=True)                           # loading alexnet

alexnet.features
alexnet.classifier

#feature_data = alexnet.features(img)                   # Feature data can be thought of as new images

# location on Google Drive
master_path = '/content/gdrive/My Drive/APS360/Lab 3/Split Data/' 

# Transform Settings - Do not use RandomResizedCrop
transform = transforms.Compose([transforms.Resize((224,224)), 
                                transforms.ToTensor()])

# As performed by code above, there are 3 folders with 60% training, 20% validation and 20% testing samples
train_dataset = torchvision.datasets.ImageFolder(master_path + 'train', transform=transform)
val_dataset = torchvision.datasets.ImageFolder(master_path + 'val', transform=transform)
test_dataset = torchvision.datasets.ImageFolder(master_path + 'test', transform=transform)

# Prepare Dataloader Again
batch_size = 27
num_workers = 1

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

train_data_iter = iter(train_loader)                 # Obtain a batch of training images to pass through Alex
tr_imgs, tr_labels = train_data_iter.next() 
train_features = alexnet.features(tr_imgs)

val_data_iter = iter(val_loader)                     # Obtain a batch of validation images to pass through Alex
val_imgs, val_labels = val_data_iter.next()
val_features = alexnet.features(val_imgs)

test_data_iter = iter(val_loader)                    # Obtain a batch of testing images to pass through Alex
test_imgs, test_labels = val_data_iter.next()
test_features = alexnet.features(test_imgs)

# Save Features to Folder 

import os
import torchvision.models
alexnet = torchvision.models.alexnet(pretrained=True)

# location on Google Drive
master_path = '/content/gdrive/My Drive/APS360/Lab 3/AlexNet' # had to manually create train, validation and testing folders

# Prepare Dataloader (requires code from 1.)
batch_size = 1                                            # save 1 file at a time, hence batch_size = 1
num_workers = 1

train_loader_1 = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                           num_workers=num_workers, shuffle=True)
val_loader_1 = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, 
                                           num_workers=num_workers, shuffle=True)
test_loader_1 = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, 
                                           num_workers=num_workers, shuffle=True)

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']

# save training features to folder as tensors
i = 0
for img, label in train_loader_1:                                               #There are 1453 items, since batch size is 1, it will iterate 1458 times
  features = alexnet.features(img)
  features_tensor = torch.from_numpy(features.detach().numpy())

  folder_name = master_path + '/train/' + str(classes[label])                   # If this path doesn't exist, make it
  if not os.path.isdir(folder_name):                                            # Keep in mind that it cannot make a folder in a folder, one a folder in the final slash
    os.mkdir(folder_name)
  torch.save(features_tensor.squeeze(0), folder_name + '/' + str(i) + '.tensor')
  i += 1

# save validation features to folder as tensors
ii = 0
for img, label in val_loader_1:
  features = alexnet.features(img)
  features_tensor = torch.from_numpy(features.detach().numpy())

  folder_name = master_path + '/validation/' + str(classes[label])              # If directory doesn't already exist, make one 
  if not os.path.isdir(folder_name):
    os.mkdir(folder_name)
  torch.save(features_tensor.squeeze(0), folder_name + '/' + str(ii) + '.tensor')
  ii += 1

# save test features to folder as tensors
iii = 0
for img, label in test_loader_1:
  features = alexnet.features(img)
  features_tensor = torch.from_numpy(features.detach().numpy())

  folder_name = master_path + '/test/' + str(classes[label])
  if not os.path.isdir(folder_name):
    os.mkdir(folder_name)
  torch.save(features_tensor.squeeze(0), folder_name + '/' + str(iii) + '.tensor')
  iii += 1

# Load Features from Folder 

master_path = '/content/gdrive/My Drive/APS360/Lab 3/AlexNet'

feature_trainset = torchvision.datasets.DatasetFolder(master_path + '/train', loader=torch.load, extensions=('.tensor'))
feature_valset = torchvision.datasets.DatasetFolder(master_path + '/validation', loader=torch.load, extensions=('.tensor'))
feature_testset = torchvision.datasets.DatasetFolder(master_path + '/test', loader=torch.load, extensions=('.tensor'))

batch_size = 27
num_workers = 1

feature_train_loader = torch.utils.data.DataLoader(feature_trainset, batch_size=batch_size, 
                                           num_workers=num_workers, shuffle=True)
feature_val_loader = torch.utils.data.DataLoader(feature_valset, batch_size=batch_size, 
                                           num_workers=num_workers, shuffle=True)
feature_test_loader = torch.utils.data.DataLoader(feature_testset, batch_size=batch_size, 
                                           num_workers=num_workers, shuffle=True)

# Verification Step - obtain one batch of features
dataiter = iter(feature_train_loader)
feature, label = dataiter.next()
print(feature.shape)                                                            # There are 256 feature maps that are 6 by 6 per image
print(label.shape)

# Creating a CNN with Transfer Learning

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim                                                     #for gradient descent
import matplotlib.pyplot as plt
import numpy as np

class CNNClassifier3(nn.Module):

    def __init__(self):
        super(CNNClassifier3, self).__init__()                                                     
        self.name = "CNN"
        self.fc1 = nn.Linear(256 * 6 * 6, 100)
        self.fc2 = nn.Linear(100, 32)                                             
        self.fc3 = nn.Linear(32, 9)                                             # 9 possible outputs


    def forward(self, x):
        x = x.view(-1, 256 * 6 * 6)                                             #flatten feature data
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

new_net = CNNClassifier3()

use_cuda = True
if use_cuda and torch.cuda.is_available():
  new_net = new_net.cuda()
  print('Using GPU')

train_acc, val_acc = train(new_net, feature_train_loader, feature_val_loader, batch_size=20, num_epochs=20, learn_rate = 0.001)

plt.plot(range(0,20), train_acc, label='Train')                                 # plotting the training accuracy
plt.plot(range(0,20), val_acc, label='Validation')                              # plotting the validation accuracy
plt.legend(loc='lower right')

# Test New Model

test_acc = get_accuracy(new_net, feature_test_loader)
print(test_acc)

# Obtaining sample images (personal images)
import os
import torchvision.models

data_path1 = '/content/gdrive/My Drive/APS360/Lab 3/Personal_Dataset'
final_test_dataset = torchvision.datasets.ImageFolder(data_path1, transform=transform)

batch_size = 1
num_workers = 1
final_test_loader = torch.utils.data.DataLoader(final_test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)


# save testing features to folder as tensors
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
features_path = '/content/gdrive/My Drive/APS360/Lab 3/Final_Test'

i = 0
for img, label in final_test_loader:                                            #There are 1453 items, since batch size is 1, it will iterate 1458 times
  features = alexnet.features(img)
  features_tensor = torch.from_numpy(features.detach().numpy())

  folder_name = features_path + '/' + str(classes[label])                       # If this path doesn't exist, make it
  if not os.path.isdir(folder_name):                                            # Keep in mind that it cannot make a folder in a folder, one a folder in the final slash
    os.mkdir(folder_name)
  torch.save(features_tensor.squeeze(0), folder_name + '/' + str(i) + '.tensor')
  i += 1

# Load Final Testing Features from Folder after passing through Alexnet

master_path1 = '/content/gdrive/My Drive/APS360/Lab 3/Final_Test'

final_feature_testset = torchvision.datasets.DatasetFolder(master_path1, loader=torch.load, extensions=('.tensor'))

batch_size = 27
num_workers = 1

final_feature_testset_loader = torch.utils.data.DataLoader(final_feature_testset, batch_size=batch_size, 
                                           num_workers=num_workers, shuffle=True)

test_acc_final = get_accuracy(new_net, final_feature_testset_loader)
print(test_acc_final)