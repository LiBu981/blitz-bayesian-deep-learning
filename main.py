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



training_dataset = MNISTDataset('C:/mnist/train-images-idx3-ubyte.gz', 'C:/mnist/train-labels-idx1-ubyte.gz')
test_dataset = MNISTDataset('C:/mnist/t10k-images-idx3-ubyte.gz', 'C:/mnist/t10k-labels-idx1-ubyte.gz')

training_batch_size = 50
test_batch_size = 1000

num_samples = 5

training_loader = torch.utils.data.DataLoader(training_dataset, batch_size = training_batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = test_batch_size, shuffle = True)

num_epochs = 1

neural_net = BayesianNet().to(device)

train_losses = []
train_counter = []
test_accuracy = []
test_losses= []

test_counter = [num * training_dataset.num_images for num in range(num_epochs + 1)]

loss_function = nn.CrossEntropyLoss()

learning_rate = .2
momentum = .9

optimizer = torch.optim.SGD(neural_net.parameters(), lr = learning_rate, momentum = momentum)

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)




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

        
    return
        

        



def sampling(image_idx):

    #predictions
    confidences = np.array([])
    predictions = np.array([])


    image, label = test_dataset.__getitem__(image_idx)
    #label.item()
    
    with torch.no_grad(): #temp. sets requires_grad=False --> no calc. of gradients; faster

        for _ in range(num_samples):
            
            output = neural_net(image)

            # confidence and prediction value (0-9) for sample
            conf_sample, pred_sample = torch.max(output, dim=0)

            confidences = np.append(confidences, conf_sample.tolist())
            predictions = np.append(predictions, pred_sample.tolist())

    #pick predictions matching label
    mask = (predictions == label.item())

    #confidences of correct predictions
    conf_corr = confidences[mask]

    #mean & std of confidences
    conf_mean = np.mean(conf_corr)
    conf_std = np.std(conf_corr)

    print('\nSampling:')
    print('Confidences in prediction: Mean: {:.2f}%  Std: {:.2e}%  Accuracy: {}/{}\n'.format(
            conf_mean*100, conf_std*100, len(conf_corr), num_samples))

    







if __name__ == "__main__":

    test()

    for epoch in range(1, num_epochs + 1):
        train(epoch)
        test()


    sampling(0)

    print('\nTotal epochs: {}'.format(num_epochs))
    print('Max Accuracy is: {}%'.format(round(100*max(test_accuracy), 2)))

    # fig = plt.figure()
    # plt.plot(train_counter, train_losses, color = 'blue', zorder = 1)
    # plt.scatter(test_counter, test_losses, color = 'red', zorder = 2)
    # plt.scatter(test_counter, test_accuracy, color = 'green', marker = '+', zorder = 3)
    # plt.legend(['Train Loss', 'Test Loss', 'Accuracy'], loc = 'upper right')
    # plt.xlabel('number of training examples seen')
    # plt.ylabel('negative log likelihood loss')
    # fig
    # plt.show()
    
