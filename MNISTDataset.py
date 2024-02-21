from torch.utils.data import Dataset
import gzip
import numpy as np
import torch
import matplotlib.pyplot as plt

class MNISTDataset(Dataset):

    def __init__(self, image_data_root, label_data_root):
            
        #Variables for the image set
        self.image_data_root = image_data_root
        self.image_magic_number = 0
        self.num_images = 0
        self.image_rows = 0
        self.image_columns = 0
        self.images = np.empty(0)
        
        #Variables for labels
        self.label_data_root = label_data_root
        self.label_magic_number = 0
        self.num_labels = 0
        self.labels = np.empty(0)

        #Functions that initialize the data
        self.image_init_dataset()
        self.label_init_dataset()

    #Returns the number of images in the set
    def __len__(self):
        return self.num_images
    
    #Returns an image based on a given index
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
    

    #This method gets the images from the MNIST dataset
    def image_init_dataset(self):

        #Unzips the image file
        image_file = gzip.open(self.image_data_root, 'r') #'r'= read as binary (not text)
    
        #Datatype that switches the byteorder for the dataset
        reorder_type = np.dtype(np.int32).newbyteorder('>')

        #Getting the first 16 bytes from the file(first 4 32-bit integers)
        self.image_magic_number = np.frombuffer(image_file.read(4), dtype=reorder_type)[0]
        self.num_images = np.frombuffer(image_file.read(4), dtype=reorder_type)[0]
        self.image_rows = np.frombuffer(image_file.read(4), dtype=reorder_type)[0]
        self.image_columns = np.frombuffer(image_file.read(4), dtype=reorder_type)[0]

        #Getting all the bytes for the images into the buffer
        buffer = image_file.read(self.num_images * self.image_rows * self.image_columns)

        #Next we read the bytes from the buffer as unsigned 8 bit integers (np.uint8), and then put them into a
        #numpy array as 32 bit floats.  This is now a 1D array (a flattened vector) of all the data
        self.images = np.frombuffer(buffer, dtype = np.uint8).astype(np.float32)
        #print(self.images.shape)

        #Here we make the 1D array into a 60000x784 array (images are flattened) to be useable with neural networks
        self.images = np.reshape(self.images, (self.num_images, 784))

        #This normalizes the data to be between 0 and 1.  The 255 is the range of the pixel values (0-255)
        self.images = self.images/255

        #Turns the data to tensors as that is the format that neural networks use
        self.images = torch.tensor(self.images)


    def label_init_dataset(self):
        #analogous to image_init_dataset

        label_file = gzip.open(self.label_data_root, 'r')
        
        reorder_type = np.dtype(np.int32).newbyteorder('>')

        self.label_magic_number = np.frombuffer(label_file.read(4), dtype=reorder_type).astype(np.int64)[0]
        self.num_labels = np.frombuffer(label_file.read(4), dtype=reorder_type).astype(np.int64)[0]
        
        buffer = label_file.read(self.num_labels)

        self.labels = np.frombuffer(buffer, dtype = np.uint8)
        
        self.labels = torch.tensor(self.labels, dtype = torch.long)





def draw_image(images_root, labels_root, image_idx):
    mnist = MNISTDataset(images_root, labels_root)

    #testing
    mnist.images = np.reshape(mnist.images, (mnist.num_images, 28, 28))
    image, label = mnist.__getitem__(image_idx)
    print('Image dimensions: {}x{}'.format(image.shape[0], image.shape[1]))
    print('Label: {}'.format(label.item()))
    plt.imshow(image)
    plt.show()

if __name__ == "__main__":
    draw_image('C:/mnist/train-images-idx3-ubyte.gz', 'C:/mnist/train-labels-idx1-ubyte.gz', 300)
    draw_image('C:/mnist/t10k-images-idx3-ubyte.gz', 'C:/mnist/t10k-labels-idx1-ubyte.gz', 300)