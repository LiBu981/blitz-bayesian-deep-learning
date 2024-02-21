from BayesianNN import BayesianNet
from MNISTDataset import MNISTDataset

import torch.nn as nn
import torch
import matplotlib.pyplot as plt

import numpy as np


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


# create train & test data
training_dataset = MNISTDataset('C:/mnist/train-images-idx3-ubyte.gz', 'C:/mnist/train-labels-idx1-ubyte.gz')
test_dataset = MNISTDataset('C:/mnist/t10k-images-idx3-ubyte.gz', 'C:/mnist/t10k-labels-idx1-ubyte.gz')


# train & test parameters
training_batch_size = 50
test_batch_size = 1000

num_epochs = 1

learning_rate = .2
momentum = .9


# load train & test data
training_loader = torch.utils.data.DataLoader(training_dataset, batch_size = training_batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = test_batch_size, shuffle = True)

# instantiate neural net
neural_net = BayesianNet().to(device)

# storage
train_losses = []
train_counter = []
test_accuracy = []
test_losses= []

# track testing (over all images & epochs)
test_counter = [num * training_dataset.num_images for num in range(num_epochs + 1)]

# loss function & optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(neural_net.parameters(), lr = learning_rate, momentum = momentum)

# seed for recreating data
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)





# train & test function

def train(epoch):

    for batch_idx, (images, labels) in enumerate(training_loader):
        
        optimizer.zero_grad()

        output = neural_net(images) #output = 50(batch size) x 10(options); options = 0-9 (for each: confidence value)

        loss = loss_function(output, labels)

        loss.backward() #calc. x.grad (gradient for each param.)
        optimizer.step() #update of param. (x += -lr * x.grad)
        
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * training_batch_size, len(training_dataset),
                    100 * batch_idx / len(training_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx * training_batch_size) + ((epoch - 1) * len(training_dataset)))


def test():

    test_loss = 0
    correct_guesses = 0

    with torch.no_grad(): #temp. sets requires_grad=False --> no calc. of gradients; faster
        for images, labels in test_loader:

            output = neural_net(images) #s.o
            
            test_loss += loss_function(output, labels).item() #sum --> only average needed later
            
            guesses = torch.max(output, 1, keepdim = True)[1] #selects option (0-9) with highest conf. value; [1]=index=0-9
            
            correct_guesses += torch.eq(guesses, labels.data.view_as(guesses)).sum() #view_as(): viewing tensors as same dim.; .sum = sum of 0 & 1 (F/T)


        test_loss /= len(test_loader.dataset)/test_batch_size #divide test_loss by amount of batches --> average
        test_losses.append(test_loss)

        current_accuracy = float(correct_guesses)/float(len(test_dataset))
        test_accuracy.append(current_accuracy)

        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct_guesses, len(test_dataset),
                100. * current_accuracy))

        

    





if __name__ == "__main__":

    # main loop (training & testing)
    test()
    for epoch in range(1, num_epochs + 1):
        train(epoch)
        test()

    print('\nTotal epochs: {}'.format(num_epochs))
    print('Max Accuracy is: {}%'.format(round(100*max(test_accuracy), 2)))


    # save trained model
    torch.save({'model_state_dict':neural_net.state_dict(),
                'num_epochs': num_epochs},
                'bayesianNN.pth')


    # plot losses & accuracy
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color = 'blue', zorder = 1)
    plt.scatter(test_counter, test_losses, color = 'red', zorder = 2)
    plt.scatter(test_counter, test_accuracy, color = 'green', marker = '+', zorder = 3)
    plt.legend(['Train Loss', 'Test Loss', 'Accuracy'], loc = 'upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    #fig
    plt.show()
