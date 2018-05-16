# AutoEncoders

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Preparing the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Creating the architecture of the Neural Network
class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_movies, 20)  #full connecting layer 1, (no. of features, no. neuron in 1st layer)
        self.fc2 = nn.Linear(20, 10)         #full connecting layer 2
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_movies)
        self.activation = nn.Sigmoid() #function available: rectifier, sigmoid
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

    
sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5) #optimizer available: RMSpro, Adam
# the () is used to iterate over all the params in sae


# Training the SAE
nb_epoch = 100
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0. #number of user that rated at least one movie
    for id_user in range(nb_users):  # 1 epoch
        input = Variable(training_set[id_user]).unsqueeze(0) # "unsqueeze(0)" create a batch of 1D index needed for the index to work
        target = input.clone() #make a copy of input
        if torch.sum(target.data > 0) > 0: #consider user that rated at least one movie
            output = sae(input)
            target.require_grad = False #optimization, don't compute the gradient with respect to the target
            output[target == 0] = 0
            loss = criterion(output, target)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10) #for that particular user, divide the total number of movies by the number of movies that has actually been rated 
            loss.backward() #tells which direction we need to update the weight, increase or decrease it
            train_loss += np.sqrt(loss.data[0]*mean_corrector) #loss is squared data so need sqrt
            s += 1.
            optimizer.step() #tells how much to update. (whereas backward tells only the direction)
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))  #expect a loss less than 1 which means 1 star

# Testing the SAE
test_loss = 0
s = 0.
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user])
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.data[0]*mean_corrector)
        s += 1.
print('test loss: '+str(test_loss/s))

####################################
########Gridsearch to fine tune
####################################


#You can change the target_user_id and target_movie_id to what you need.
xxx = pd.read_csv('ml-100k/u2.base', delimiter = '\t')
trial_set = pd.read_csv('ml-100k/u2.base', delimiter = '\t')
trial_set = np.array(trial_set, dtype = 'int')
trial_set = convert(trial_set)

trial_set1 = torch.FloatTensor(trial_set)

for user_id in range(nb_users):
    target_user_id = user_id
    target_movie_id = 1
    input = Variable(trial_set1[target_user_id-1]).unsqueeze(0)
    #target = input.clone()
    if input.data == 0:
        output = sae(input)
        output_numpy = output.data.numpy()
print (''+ str(output_numpy[0,target_movie_id-1]))
#    loss = criterion(output, input)


#Testing how well the model works on u2 base dataset
test_loss = 0
s_test = 0.
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = Variable(trial_set1[id_user])
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.data[0]*mean_corrector)
        s += 1.
print('test loss: '+str(test_loss/s))





# Testing the SAE
test_loss = 0
s_test = 0.
output_numpy_all = np.empty([0,nb_movies])
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user])
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0
        output_numpy = output.data.numpy()
        output_numpy_all = np.vstack((output_numpy_all, output_numpy) )
        loss = criterion(output, target)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.data[0]*mean_corrector)
        s_test += 1.