import numpy as np
import matplotlib.pyplot as plt
import torch

from BayesianNN import BayesianNet
from MNISTDataset import MNISTDataset

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# create test data
test_dataset = MNISTDataset('C:/mnist/t10k-images-idx3-ubyte.gz', 'C:/mnist/t10k-labels-idx1-ubyte.gz')

# instantiate neural net
neural_net = BayesianNet().to(device)

# load saved net & training parameters
saved_model = torch.load('bayesianNN.pth')
neural_net.load_state_dict(saved_model['model_state_dict'])

training_batch_size = saved_model['training_batch_size']
num_epochs = saved_model['num_epochs']
learning_rate = saved_model['learning_rate']
momentum = saved_model['momentum']


# number of samples / image
num_samples = 20


# Sampling
numb_value = 0 #0-9

# storage
conf_mean_all = []
conf_std_all = []


def sampling(image_idx):

    #predictions
    confidences = np.array([])
    predictions = np.array([])


    image, label = test_dataset.__getitem__(image_idx)
    
    with torch.no_grad(): # temp. sets requires_grad=False --> no calc. of gradients; faster

        for _ in range(num_samples):
            
            output = neural_net(image)

            # confidence and prediction value (0-9) for sample
            conf_sample, pred_sample = torch.max(output, dim=0)

            confidences = np.append(confidences, conf_sample.tolist())
            predictions = np.append(predictions, pred_sample.tolist())

    # pick predictions matching label
    mask = (predictions == label.item())

    # confidences of correct predictions
    conf_corr = confidences[mask]

    # mean & std of confidences
    conf_mean = np.mean(conf_corr)
    conf_std = np.std(conf_corr)
    
    return conf_mean, conf_std


# counter for labels matching number value (0-9)
label_match = 0

print('searching for images with label = {} in test data'.format(numb_value))

for image_idx in range(len(test_dataset)):

    label = test_dataset.__getitem__(image_idx)[1]

    if label.item() == numb_value:
        conf_mean, conf_std = sampling(image_idx)
        conf_mean_all.append(conf_mean)
        conf_std_all.append(conf_std)

        label_match += 1
    
    if image_idx % 500 == 0:
        print('images scanned: {}/{} ({:.0f}%); images with label={}: {}'.format(
            image_idx, len(test_dataset), image_idx/len(test_dataset)*100, numb_value, label_match))


# plotting confindence (mean & std) in subplots
fig = plt.figure()
plt.subplot(2,1,1)
plt.hist(conf_mean_all,bins=np.linspace(0.9,1,100))
plt.xlabel('mean of confidences in prediction ({} samples / image)'.format(num_samples))
plt.ylabel('total number of occurences')

plt.subplot(2,1,2)
plt.hist(conf_std_all,bins=np.linspace(0.9,1,100))
plt.xlabel('std of confidence in prediction ({} samples / image)'.format(num_samples))
plt.ylabel('total number of occurences')

fig.suptitle('label: {}'.format(numb_value), fontsize=15)
fig.text(0.5, 0.95, 'batch size: {}, number of epochs: {}, learning rate: {}, momentum: {}'.format(
        training_batch_size, num_epochs, learning_rate, momentum),
        ha='center', va='top', fontsize=10)

plt.show()




# print('\nSampling:')
# print('Confidences in prediction: Mean: {:.2f}%  Std: {:.2e}%  Accuracy: {}/{}\n'.format(
#         conf_mean*100, conf_std*100, len(conf_corr), num_samples))