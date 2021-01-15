import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms


##############################################################################
#Data Loading and Helper Functions

def get_relevant_indices(dataset, classes, target_classes):
    """ Return the indices for datapoints in the dataset that belongs to the
    desired target classes, a subset of all possible classes.

    Args:
        dataset: Dataset object
        classes: A list of strings denoting the name of each class
        target_classes: A list of strings denoting the name of desired classes
                        Should be a subset of the 'classes'
    Returns:
        indices: list of indices that have labels corresponding to one of the
                 target classes
    """
    indices = []
    for i in range(len(dataset)):
        # Check if the label is in the target classes
        label_index = dataset[i][1] # ex: 3
        label_class = classes[label_index] # ex: 'cat'
        if label_class in target_classes:
            indices.append(i)
    return indices

def get_data_loader(target_classes, batch_size):
    """ Loads images of cats and dogs, splits the data into training, validation
    and testing datasets. Returns data loaders for the three preprocessed datasets.

    Args:
        target_classes: A list of strings denoting the name of the desired
                        classes. Should be a subset of the argument 'classes'
        batch_size: A int representing the number of samples per batch
    
    Returns:
        train_loader: iterable training dataset organized according to batch size
        val_loader: iterable validation dataset organized according to batch size
        test_loader: iterable testing dataset organized according to batch size
        classes: A list of strings denoting the name of each class
    """

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    ########################################################################
    # The output of torchvision datasets are PILImage images of range [0, 1].
    # We transform them to Tensors of normalized range [-1, 1].
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # Load CIFAR10 training data
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    # Get the list of indices to sample from
    relevant_indices = get_relevant_indices(trainset, classes, target_classes)
    
    # Split into train and validation
    np.random.seed(1000) # Fixed numpy random seed for reproducible shuffling
    np.random.shuffle(relevant_indices)
    split = int(len(relevant_indices) * 0.8) #split at 80%
    
    # split into training and validation indices
    relevant_train_indices, relevant_val_indices = relevant_indices[:split], relevant_indices[split:]  
    train_sampler = SubsetRandomSampler(relevant_train_indices)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               num_workers=1, sampler=train_sampler)
    val_sampler = SubsetRandomSampler(relevant_val_indices)
    val_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              num_workers=1, sampler=val_sampler)
    # Load CIFAR10 testing data
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    # Get the list of indices to sample from
    relevant_test_indices = get_relevant_indices(testset, classes, target_classes)
    test_sampler = SubsetRandomSampler(relevant_test_indices)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             num_workers=1, sampler=test_sampler)
    return train_loader, val_loader, test_loader, classes

###############################################################################
# Training
def get_model_name(name, batch_size, learning_rate, epoch):
    """ Generate a name for the model consisting of all the hyperparameter values

    Args:
        config: Configuration object containing the hyperparameters
    Returns:
        path: A string with the hyperparameter name and value concatenated
    """
    path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(name,
                                                   batch_size,
                                                   learning_rate,
                                                   epoch)
    return path

def normalize_label(labels):
    """
    Given a tensor containing 2 possible values, normalize this to 0/1

    Args:
        labels: a 1D tensor containing two possible scalar values
    Returns:
        A tensor normalize to 0/1 value
    """
    max_val = torch.max(labels)
    min_val = torch.min(labels)
    norm_labels = (labels - min_val)//(max_val - min_val)
    return norm_labels

def evaluate(net, loader, criterion):
    """ Evaluate the network on the validation set.

     Args:
         net: PyTorch neural network object
         loader: PyTorch data loader for the validation set
         criterion: The loss function
     Returns:
         err: A scalar for the avg classification error over the validation set
         loss: A scalar for the average loss function over the validation set
     """
    total_loss = 0.0
    total_err = 0.0
    total_epoch = 0
    for i, data in enumerate(loader, 0):
        inputs, labels = data
        labels = normalize_label(labels)  # Convert labels to 0/1
        outputs = net(inputs)
        loss = criterion(outputs, labels.float())
        corr = (outputs > 0.0).squeeze().long() != labels
        total_err += int(corr.sum())
        total_loss += loss.item()
        total_epoch += len(labels)
    err = float(total_err) / total_epoch
    loss = float(total_loss) / (i + 1)
    return err, loss

###############################################################################
# Training Curve
def plot_training_curve(path):
    """ Plots the training curve for a model run, given the csv files
    containing the train/validation error/loss.

    Args:
        path: The base path of the csv files produced during training
    """
    import matplotlib.pyplot as plt
    train_err = np.loadtxt("{}_train_err.csv".format(path))
    val_err = np.loadtxt("{}_val_err.csv".format(path))
    train_loss = np.loadtxt("{}_train_loss.csv".format(path))
    val_loss = np.loadtxt("{}_val_loss.csv".format(path))
    plt.title("Train vs Validation Error")
    n = len(train_err) # number of epochs
    plt.plot(range(1,n+1), train_err, label="Train")
    plt.plot(range(1,n+1), val_err, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend(loc='best')
    plt.show()
    plt.title("Train vs Validation Loss")
    plt.plot(range(1,n+1), train_loss, label="Train")
    plt.plot(range(1,n+1), val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()


# Model Architecture 

class LargeNet(nn.Module):
    def __init__(self):
        super(LargeNet, self).__init__()
        self.name = "large"
        self.conv1 = nn.Conv2d(3, 5, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(5, 10, 5)
        self.fc1 = nn.Linear(10 * 5 * 5, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 10 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.squeeze(1) # Flatten to [batch_size]
        return x

class SmallNet(nn.Module):
    def __init__(self):
        super(SmallNet, self).__init__()
        self.name = "small"
        self.conv = nn.Conv2d(3, 5, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(5 * 7 * 7, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = self.pool(x)
        x = x.view(-1, 5 * 7 * 7)
        x = self.fc(x)
        x = x.squeeze(1) # Flatten to [batch_size]
        return x


# Training Code
def train_net(net, batch_size=64, learning_rate=0.01, num_epochs=30):
    ########################################################################
    # Train a classifier on cats vs dogs
    target_classes = ["cat", "dog"]
    ########################################################################
    # Fixed PyTorch random seed for reproducible result
    torch.manual_seed(1000)
    ########################################################################
    # Obtain the PyTorch data loader objects to load batches of the datasets
    train_loader, val_loader, test_loader, classes = get_data_loader(
            target_classes, batch_size)
    ########################################################################
    # Define the Loss function and optimizer
    # The loss function will be Binary Cross Entropy (BCE). In this case we
    # will use the BCEWithLogitsLoss which takes unnormalized output from
    # the neural network and scalar label.
    # Optimizer will be SGD with Momentum.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    ########################################################################
    # Set up some numpy arrays to store the training/test loss/erruracy
    train_err = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)
    val_err = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)
    ########################################################################
    # Train the network
    # Loop over the data iterator and sample a new batch of training data
    # Get the output from the network, and optimize our loss function.
    start_time = time.time()
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        total_train_loss = 0.0
        total_train_err = 0.0
        total_epoch = 0
        for i, data in enumerate(train_loader, 0):
            # Get the inputs
            inputs, labels = data
            labels = normalize_label(labels) # Convert labels to 0/1
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass, backward pass, and optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            # Calculate the statistics
            corr = (outputs > 0.0).squeeze().long() != labels
            total_train_err += int(corr.sum())
            total_train_loss += loss.item()
            total_epoch += len(labels)
        train_err[epoch] = float(total_train_err) / total_epoch
        train_loss[epoch] = float(total_train_loss) / (i+1)
        val_err[epoch], val_loss[epoch] = evaluate(net, val_loader, criterion)
        print(("Epoch {}: Train err: {}, Train loss: {} |"+
               "Validation err: {}, Validation loss: {}").format(
                   epoch + 1,
                   train_err[epoch],
                   train_loss[epoch],
                   val_err[epoch],
                   val_loss[epoch]))
        # Save the current model (checkpoint) to a file
        model_path = get_model_name(net.name, batch_size, learning_rate, epoch)
        torch.save(net.state_dict(), model_path)
    print('Finished Training')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Total time elapsed: {:.2f} seconds".format(elapsed_time))
    # Write the train/test loss/err into CSV file for plotting later
    epochs = np.arange(1, num_epochs + 1)
    np.savetxt("{}_train_err.csv".format(model_path), train_err)
    np.savetxt("{}_train_loss.csv".format(model_path), train_loss)
    np.savetxt("{}_val_err.csv".format(model_path), val_err)
    np.savetxt("{}_val_loss.csv".format(model_path), val_loss)


# Since the function writes files to disk, you will need to mount
# your Google Drive. If you are working on the lab locally, you
# can comment out this code.

from google.colab import drive
drive.mount('/content/gdrive')

train_net(small_net , batch_size=64, learning_rate=0.01, num_epochs=30)

train_net(large_net , batch_size=64, learning_rate=0.01, num_epochs=30)

"""####Answer to part (d)

The small_net took less time than the large_net. It makes sense that the smaller net took less time because it has few parameters to change and worry about, the small_net has 386 parameters and the large_net has 9705 parameters (significantly more for the large_net).

### Part (e) - 2pt

Use the function `plot_training_curve` to display the trajectory of the 
training/validation error and the training/validation loss.
You will need to use the function `get_model_name` to generate the
argument to the `plot_training_curve` function.

Do this for both the small network and the large network. Include both plots
in your writeup.
"""

model_path_small = get_model_name("small", batch_size=64, learning_rate=0.01, epoch=29)                 #initialize a name for the model
plot_training_curve(model_path_small)

model_path_large = get_model_name("large", batch_size=64, learning_rate=0.01, epoch=29) 
plot_training_curve(model_path_large)

large_net = LargeNet()                                                                          #re-constructing model
train_net(large_net , batch_size=64, learning_rate=0.001, num_epochs=30)                        #training the model

model_path_large = get_model_name("large", batch_size=64, learning_rate=0.001, epoch=29)        #plotting the results
plot_training_curve(model_path_large)


large_net = LargeNet()                                                                        #re-constructing model
train_net(large_net , batch_size=64, learning_rate=0.1, num_epochs=30)                        #training the model

model_path_large = get_model_name("large", batch_size=64, learning_rate=0.1, epoch=29)        #plotting the results
plot_training_curve(model_path_large)


large_net = LargeNet()                                                                        #re-constructing model
train_net(large_net , batch_size=512, learning_rate=0.01, num_epochs=30)                        #training the model

model_path_large = get_model_name("large", batch_size=512, learning_rate=0.01, epoch=29)        #plotting the results
plot_training_curve(model_path_large)


large_net = LargeNet()                                                                        #re-constructing model
train_net(large_net , batch_size=16, learning_rate=0.01, num_epochs=30)                        #training the model

model_path_large = get_model_name("large", batch_size=16, learning_rate=0.01, epoch=29)        #plotting the results
plot_training_curve(model_path_large)

# Use Best Hyperparameters

# Note: When we re-construct the model, we start the training
# with *random weights*. If we omit this code, the values of
# the weights will still be the previously trained values.
large_net = LargeNet()                                                                           #re-constructing model
train_net(large_net , batch_size=512, learning_rate=0.001, num_epochs=30)                        #training the model

model_path_large = get_model_name("large", batch_size=512, learning_rate=0.001, epoch=29)        #plotting the results
plot_training_curve(model_path_large)



large_net = LargeNet()                                                                          #re-constructing model
train_net(large_net , batch_size=512, learning_rate=0.01, num_epochs=50)                        #training the model

model_path_large = get_model_name("large", batch_size=512, learning_rate=0.01, epoch=49)        #plotting the results
plot_training_curve(model_path_large)


net = LargeNet()                                                                          #Initialing a large network 
model_path = get_model_name("large", batch_size=512, learning_rate=0.01, epoch=50)
state = torch.load(model_path_large)
net.load_state_dict(state)

train_loader, val_loader, test_loader, classes = get_data_loader(
    target_classes=["cat", "dog"], 
    batch_size=64)

def test_net(net, batch_size=64, learning_rate=0.01, num_epochs=30):
    ########################################################################
    # Fixed PyTorch random seed for reproducible result
    torch.manual_seed(1000)
    ########################################################################
    # Define the Loss function and optimizer
    # The loss function will be Binary Cross Entropy (BCE). In this case we
    # will use the BCEWithLogitsLoss which takes unnormalized output from
    # the neural network and scalar label.
    # Optimizer will be SGD with Momentum.
    criterion = nn.BCEWithLogitsLoss()
    ########################################################################
    # Set up some numpy arrays to store the test and validation error
    test_err = np.zeros(num_epochs)
    val_err = np.zeros(num_epochs)
    ########################################################################
    # Train the network
    # Loop over the data iterator and sample a new batch of training data
    # Get the output from the network, and optimize our loss function.
    start_time = time.time()
    total_test_err = 0.0
    total_epoch = 0
    #Start calculating error
    for i, data in enumerate(test_loader, 0):                                #loop over the testing set as many times as there are items
        # Get the inputs
        inputs, labels = data                                                #retrieve the inputs and the correct label
        labels = normalize_label(labels)                                     #Convert labels to either 0,1 (helper function)

        #getting the outputs after passing them through CNN
        outputs = net(inputs)                                               

        # Calculate the statistics
        corr = (outputs > 0.0).squeeze().long() != labels
        total_test_err += int(corr.sum())                                    #finds number of incorrect predictions

        total_epoch += len(labels)                                           #finds the epoch (length of dataset)

    test_err = float(total_test_err) / total_epoch                           #calculates percentage of error                     
    val_err, _ = evaluate(net, val_loader, criterion)                        #finds validation error and loss using evaluate function
    print(("\nTest err: {}, Validation err: {}").format(
                test_err,
                val_err))
    
    # Save the current model (checkpoint) to a file
    print('\nFinished Training')                                          
    end_time = time.time()                                                    #stop timer and print total time elapsed
    elapsed_time = end_time - start_time
    print("Total time elapsed: {:.2f} seconds".format(elapsed_time))


#Test Net
test_net(net)

##########################################################################################
# Comparison to ANN

#Defining a simple two layer artificial neural network
class Pigeon(nn.Module):
    def __init__(self):
        super(Pigeon, self).__init__()
        self.name = "pigeon"
        self.layer1 = nn.Linear(32 * 32 * 3, 30)                                 #considered greyscale to RGB (1 channel to 3 channels)
        self.layer2 = nn.Linear(30, 1)
    def forward(self, img):
        flattened = img.view(-1, 32 * 32 *3)
        activation1 = self.layer1(flattened)
        activation1 = F.relu(activation1)
        activation2 = self.layer2(activation1)
        activation2 = activation2.squeeze(1)
        return activation2

#Create an ANN model 
ann = Pigeon()

#Loading the datasets
train_loader, val_loader, test_loader, classes = get_data_loader(                #train_loader, val_loader and test_loader are all datasets
    target_classes=["cat", "dog"], 
    batch_size=64) # One image per batch


# Training  Code
def train_net(ann, batch_size=64, learning_rate=0.01, num_epochs=30):
    ########################################################################
    # Train a classifier on cats vs dogs
    target_classes = ["cat", "dog"]
    ########################################################################
    # Fixed PyTorch random seed for reproducible result
    torch.manual_seed(1000)
    ########################################################################
    # Define the Loss function and optimizer
    # The loss function will be Binary Cross Entropy (BCE). In this case we
    # will use the BCEWithLogitsLoss which takes unnormalized output from
    # the neural network and scalar label.
    # Optimizer will be SGD with Momentum.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(ann.parameters(), lr=learning_rate, momentum=0.9)
    ########################################################################
    # Set up some numpy arrays to store the training/test loss/erruracy
    train_err = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)
    val_err = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)
    ########################################################################
    # Train the network
    # Loop over the data iterator and sample a new batch of training data
    # Get the output from the network, and optimize our loss function.
    start_time = time.time()
    for epoch in range(num_epochs):                                              #loop over the dataset as many times are there are epochs
        total_train_loss = 0.0                                                   #reset training loss, error and total epoch
        total_train_err = 0.0
        total_epoch = 0
        for i, data in enumerate(train_loader, 0):                               #loop over the training set as many times as there are items
            # Get the inputs
            inputs, labels = data                                                #retrieve the inputs and the correct label
            labels = normalize_label(labels)                                     #Convert labels to either 0,1 (helper function)
            # Zero the parameter gradients
            optimizer.zero_grad()                                                #
            # Forward pass, backward pass, and optimize
            outputs = ann(inputs)                                                #getting the outputs after passing them through CNN
            loss = criterion(outputs, labels.float())                            #calculate loss using loss function
            loss.backward()                                                      #accumulates the gradient 
            optimizer.step()                                                     #changes the weights based on gradient and learning rate, momentum
            # Calculate the statistics
            corr = (outputs > 0.0).squeeze().long() != labels
            total_train_err += int(corr.sum())                                   #finds number of incorrect predictions
            total_train_loss += loss.item()                                      #finds total loss
            total_epoch += len(labels)                                           #finds the epoch (length of dataset)
        train_err[epoch] = float(total_train_err) / total_epoch                  #calculates percentage of error
        train_loss[epoch] = float(total_train_loss) / (i+1)                      
        val_err[epoch], val_loss[epoch] = evaluate(ann, val_loader, criterion)   #finds validation error and loss using evaluate function
        print(("Epoch {}: Train err: {}, Train loss: {} |"+
               "Validation err: {}, Validation loss: {}").format(
                   epoch + 1,
                   train_err[epoch],
                   train_loss[epoch],
                   val_err[epoch],
                   val_loss[epoch]))
        # Save the current model (checkpoint) to a file
        model_path = get_model_name(ann.name, batch_size, learning_rate, epoch)  #naming this NN, this NN has updated parameters, saves a model per epoch with a slightly different name, will not have csv files 
        torch.save(ann.state_dict(), model_path)                                 #save this NN to somewhere by file name
    print('Finished Training')                                          
    end_time = time.time()                                                       #stop timer and print total time elapsed
    elapsed_time = end_time - start_time
    print("Total time elapsed: {:.2f} seconds".format(elapsed_time))
    # Write the train/test loss/err into CSV file for plotting later
    epochs = np.arange(1, num_epochs + 1)
    np.savetxt("{}_train_err.csv".format(model_path), train_err)                 #saving csv to last model_path which is named with final epoch (epoch starts at 0, so after 30 iterations, the last epoch number will be 29)
    np.savetxt("{}_train_loss.csv".format(model_path), train_loss)
    np.savetxt("{}_val_err.csv".format(model_path), val_err)
    np.savetxt("{}_val_loss.csv".format(model_path), val_loss)

train_net(ann, batch_size=64, learning_rate=0.01, num_epochs=30)
model_path = get_model_name("pigeon", batch_size=64, learning_rate=0.01, epoch=29)        #plotting the results
plot_training_curve(model_path)

# Testing Code
def test_net(net, batch_size=64, learning_rate=0.01, num_epochs=30):
    ########################################################################
    # Fixed PyTorch random seed for reproducible result
    torch.manual_seed(1000)
    ########################################################################
    # Define the Loss function and optimizer
    # The loss function will be Binary Cross Entropy (BCE). In this case we
    # will use the BCEWithLogitsLoss which takes unnormalized output from
    # the neural network and scalar label.
    # Optimizer will be SGD with Momentum.
    criterion = nn.BCEWithLogitsLoss()
    ########################################################################
    # Set up some numpy arrays to store the test and validation error
    test_err = np.zeros(num_epochs)
    val_err = np.zeros(num_epochs)
    ########################################################################
    # Train the network
    # Loop over the data iterator and sample a new batch of training data
    # Get the output from the network, and optimize our loss function.
    start_time = time.time()
    total_test_err = 0.0
    total_epoch = 0
    #Start calculating error
    for i, data in enumerate(test_loader, 0):                                #loop over the testing set as many times as there are items
        # Get the inputs
        inputs, labels = data                                                #retrieve the inputs and the correct label
        labels = normalize_label(labels)                                     #Convert labels to either 0,1 (helper function)

        #getting the outputs after passing them through CNN
        outputs = net(inputs)                                               

        # Calculate the statistics
        corr = (outputs > 0.0).squeeze().long() != labels
        total_test_err += int(corr.sum())                                    #finds number of incorrect predictions

        total_epoch += len(labels)                                           #finds the epoch (length of dataset)

    test_err = float(total_test_err) / total_epoch                           #calculates percentage of error                     
    val_err, _ = evaluate(net, val_loader, criterion)                        #finds validation error and loss using evaluate function
    print(("\nTest err: {}, Validation err: {}").format(
                test_err,
                val_err))
    
    # Save the current model (checkpoint) to a file
    print('\nFinished Training')                                          
    end_time = time.time()                                                    #stop timer and print total time elapsed
    elapsed_time = end_time - start_time
    print("Total time elapsed: {:.2f} seconds".format(elapsed_time))


#Test Net
test_net(ann)
