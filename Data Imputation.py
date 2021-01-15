import csv
import numpy as np
import random
import torch
import torch.utils.data

import pandas as pd

header = ['age', 'work', 'fnlwgt', 'edu', 'yredu', 'marriage', 'occupation',
 'relationship', 'race', 'sex', 'capgain', 'caploss', 'workhr', 'country']
df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    names=header,
    index_col=False)

df.shape # there are 32561 rows (records) in the data frame, and 14 columns (features)

df[0:3] # show the first 3 records (show rows 0,1,2)

subdf = df[["age", "yredu", "capgain", "caploss", "workhr"]]
subdf[:3] # show the first 3 records in just the columns specified by subdf

np.sum(subdf["caploss"])

max_val = np.max(subdf)
min_val = np.min(subdf)
avg_val = np.mean(subdf)
print("Maximum values")
print(max_val)
print("Minimum values")
print(min_val)
print("Average values")
print(avg_val)

header = ['age', 'work', 'fnlwgt', 'edu', 'yredu', 'marriage', 'occupation',
 'relationship', 'race', 'sex', 'capgain', 'caploss', 'workhr', 'country']
df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    names=header,
    index_col=False)

df["age"] = df["age"]/max_val.age
df["yredu"] = df["yredu"]/max_val.yredu
df["capgain"] = df["capgain"]/max_val.capgain
df["caploss"] = df["caploss"]/max_val.caploss
df["workhr"] = df["workhr"]/max_val.workhr
print(df[["age", "yredu", "capgain", "caploss", "workhr"]])

numMales = sum(df["sex"] == " Male")
numFemales = sum(df["sex"] == " Female")

percent_female = (numFemales/(numFemales+numMales)*100)

print(round(percent_female,2), '% females')

contcols = ["age", "yredu", "capgain", "caploss", "workhr"]                     # number entries
catcols = ["work", "marriage", "occupation", "edu", "relationship", "sex"]      # categorical entries 
features = contcols + catcols
df = df[features]

missing = pd.concat([df[c] == " ?" for c in catcols], axis=1).any(axis=1)       # finding all rows with missing values
df_with_missing = df[missing]
df_not_missing = df[~missing]                                                   # not missing rows

missing = (len(df_with_missing))
not_missing = (len(df_not_missing))


print(missing)
percent_removed = 100*(missing/(missing+not_missing))
print(round(percent_removed,2), '% removed')

data = pd.get_dummies(df_not_missing)

data[:10]

num_columns =len(data.columns)
print(num_columns)

# set the numpy seed for reproducibility
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.seed.html
np.random.seed(50)
np.random.shuffle(datanp)

rows = datanp.shape[0]
train = int(rows*0.7)
val = int(rows*0.85)

train_set = datanp[:train]
validation_set = datanp[train:val]
test_set = datanp[val:]

print(train_set.shape[0])
print(validation_set.shape[0])
print(test_set.shape[0])

# Autoencoder Setup
from torch import nn

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(57, 40),
            nn.ReLU(),
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Linear(20, 11),
            nn.ReLU()

        )
        self.decoder = nn.Sequential(
            nn.Linear(11, 20), 
            nn.ReLU(),
            nn.Linear(20, 40),
            nn.ReLU(),
            nn.Linear(40, 57),
            nn.Sigmoid()                    # get to the range (0, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

import time
import torch
import matplotlib.pyplot as plt

def zero_out_feature(records, feature):
    """ Set the feature missing in records, by setting the appropriate
    columns of records to 0
    """
    start_index = cat_index[feature]
    stop_index = cat_index[feature] + len(cat_values[feature])
    records[:, start_index:stop_index] = 0                                      # All rows set to 0 from start to stop index
    return records

def zero_out_random_feature(records):
    """ Set one random feature missing in records, by setting the 
    appropriate columns of records to 0
    """
    return zero_out_feature(records, random.choice(catcols))                    # Choose random feature to zero out 

def train(model, train_loader, valid_loader, num_epochs=5, learning_rate=1e-4):

    start_time = time.time()
    print ("Training Started...")
    

    torch.manual_seed(42)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Store loss and accuracy in lists for each epoch 
    train_acc, val_acc, train_loss, val_loss = [],[],[],[]

    for epoch in range(num_epochs):
        
        total_train_loss = 0.0
        total_val_loss = 0.0
        n = 1
        i = 1
        for data in train_loader:

            if use_cuda and torch.cuda.is_available():
                data = data.cuda()                                              # send data to cuda
            

            datam = zero_out_random_feature(data.clone())                       # zero out one categorical feature on a cloned item
            recon = model(datam)                                                # pass through model
            loss = criterion(recon, data)                                       # compare ground truth with prediction
            loss.backward()                                                     # back propagation
            optimizer.step()                                                    # update weights
            optimizer.zero_grad()                                               # zero gradient
            total_train_loss += loss.item()                   
            n = n+1

        # tracking validation loss
        train_loss.append(total_train_loss/n)     

        for data in valid_loader:
            if use_cuda and torch.cuda.is_available():
                data = data.cuda()                                              # send data to cuda

            datam = zero_out_random_feature(data.clone())                       # zero out one categorical feature on a cloned item
            recon = model(datam)                                                # pass through model
            loss = criterion(recon, data)                                       # compare ground truth with prediction
            total_val_loss += loss.item()
            i = i+1

        val_loss.append(total_val_loss/i)
        
        # tracking accuracy
        train_acc.append(get_accuracy(model, train_loader))
        val_acc.append(get_accuracy(model, valid_loader))

        print(epoch, train_loss[-1], val_loss[-1],train_acc[-1], val_acc[-1])

    n = len(train_acc)

    print('Finished Training')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Total time elapsed: {:.2f} seconds".format(elapsed_time))

    # plot loss
    plt.title("Epoch vs Loss")    
    plt.plot(range(0,n), train_loss, label='Train')                             # plotting the training accuracy
    plt.plot(range(0,n), val_loss, label='Validation')                          # plotting the validation accuracy
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='upper right')
    plt.show()
    print("Final Train Loss: {}".format(train_loss[-1]))
    print("Final Validation Loss: {}".format(val_loss[-1]))


    # plot accuracy
    plt.title("Epoch vs Accuracy")    
    plt.plot(range(0,n), train_acc, label='Train')                              # plotting the training accuracy
    plt.plot(range(0,n), val_acc, label='Validation')                           # plotting the validation accuracy
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='lower right')
    plt.show()
    print("Final Train Accuracy: {}".format(train_acc[-1]))
    print("Final Validation Accuracy: {}".format(val_acc[-1]))

# Trial 1

use_cuda = True

batch_size = 64
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=1, shuffle=True)
valid_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, num_workers=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=1, shuffle=True)

ae = AutoEncoder()

if use_cuda and torch.cuda.is_available():
    ae = ae.cuda()
    print("Training on GPU...")

train(ae, train_loader, valid_loader, num_epochs=30, learning_rate=0.001)

# Trial 2

use_cuda = True

batch_size = 64
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=1, shuffle=True)
valid_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, num_workers=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=1, shuffle=True)

test1 = AutoEncoder()

if use_cuda and torch.cuda.is_available():
    test1 = test1.cuda()
    print("Training on GPU...")

train(test1, train_loader, valid_loader, num_epochs=30, learning_rate=0.0001)

# Trial 3

use_cuda = True

batch_size = 128
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=1, shuffle=True)
valid_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, num_workers=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=1, shuffle=True)

test2 = AutoEncoder()

if use_cuda and torch.cuda.is_available():
    test2 = test2.cuda()
    print("Training on GPU...")

train(test2, train_loader, valid_loader, num_epochs=30, learning_rate=0.001)

# Trial 4

use_cuda = True

batch_size = 64
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=1, shuffle=True)
valid_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, num_workers=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=1, shuffle=True)

test3 = AutoEncoder()

if use_cuda and torch.cuda.is_available():
    test3 = test3.cuda()
    print("Training on GPU...")

train(test3, train_loader, valid_loader, num_epochs=40, learning_rate=0.001)

# Trial 5

use_cuda = True

batch_size = 64
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=1, shuffle=True)
valid_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, num_workers=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=1, shuffle=True)

test4 = AutoEncoder()

if use_cuda and torch.cuda.is_available():
    test4 = test4.cuda()
    print("Training on GPU...")

train(test4, train_loader, valid_loader, num_epochs=50, learning_rate=0.1)

# Test Loader

test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=1, shuffle=True)

test_acc = get_accuracy(ae, test_loader)
print(test_acc)

# Baseline Model 

contcols = ["age", "yredu", "capgain", "caploss", "workhr"]                     # number entries
catcols = ["work", "marriage", "occupation", "edu", "relationship", "sex"]      # categorical entries
rows = df_not_missing.shape[0]                                                  # There are 30718 rows in df_not_missing
train_idx = int(rows*0.7)
val_idx = int(rows*0.85)
                              
base_trainset = df_not_missing[:train_idx]                                      # Base trainset to get modes of each feature 
base_trainset = base_trainset.drop(contcols, axis=1)                            # dropping the contcols from train set

mode = base_trainset.mode().values.tolist()
mode = mode[0]                                                                  # mode is a list of most common entries in each category


test_set = df_not_missing[val_idx:]
test_set = test_set.drop(contcols, axis=1)                                      # dropping the contcols from testset
test_set = test_set.values.tolist()                                             

num_corr = 0
n=0
for cat in mode:
    for i in test_set:
        if cat == i[n]:
            num_corr+=1
    n+=1

total_num_entries = len(test_set)*len(test_set[0])

test_acc = num_corr/total_num_entries

print(test_acc)

# Predicting Education Level from Other Features

test_set1 = test_set[0]
test_set1 = torch.from_numpy(test_set1)

zeroed = test_set1.clone()

start_index = cat_index["edu"]
stop_index = start_index + len(cat_values["edu"])
zeroed[start_index:stop_index] = 0                      # All rows set to 0 from start to stop index

out = test1(zeroed)

prediction = get_feature(out.detach().numpy(), "edu")
actual = get_feature(test_set1.detach().numpy(), "edu")

print("Prediction: " + prediction)
print("Actual: " + actual)

# Baseline Model Prediction
print(mode[3])