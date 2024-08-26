import torch # type: ignore
from torch import nn # type: ignore
from tqdm.auto import tqdm # type: ignore
from torchvision import transforms # type: ignore
from torchvision.datasets import MNIST # type: ignore
from torchvision.utils import make_grid # type: ignore
from torch.utils.data import DataLoader # type: ignore
import matplotlib.pyplot as plt # type: ignore
torch.manual_seed(0) # Set for our testing purposes, please do not change!

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28), nrow=5, show=True):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    fig = plt.figure(figsize=(7,7))
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=nrow)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    if show:
        plt.show()
    return fig

def get_noise(n_samples, input_dim, device=torch.device("mps")):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, input_dim)
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        input_dim: the dimension of the input vector, a scalar
        device: the device type
    '''
    return torch.randn(n_samples, input_dim, device=device)


# UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_one_hot_labels
import torch.nn.functional as F # type: ignore
def get_one_hot_labels(labels, n_classes):
    """
    Function for creating one-hot vectors for the labels, returns a tensor of shape (batch_size, num_classes).
    
    Parameters:
        labels: tensor of labels from the dataloader, size (batch_size,)
        n_classes: the total number of classes in the dataset, an integer scalar
    
    Returns:
        A tensor of shape (batch_size, num_classes) with one-hot encoding.
    """
    # # Ensure labels are squeezed to remove any singleton dimensions
    # labels = labels.squeeze()
    
    # # Create the one-hot encoded tensor
    # one_hot = torch.zeros(labels.size(0), n_classes, device=labels.device)
    
    # # Scatter 1s in the appropriate positions
    # one_hot.scatter_(1, labels.unsqueeze(1), 1)
    # x_labels = labels.squeeze(dim=0)
    # one_hot = torch.zeros(x_labels.shape[0],n_classes)
    # one_hot[range(x_labels.shape[0]),x_labels[range(x_labels.shape[0])]] = 1
    # return one_hot.unsqueeze(0).to(int)
    return F.one_hot(labels,n_classes)

# UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: combine_vectors
def combine_vectors(x, y):
    '''
    Function for combining two vectors with shapes (n_samples, ?) and (n_samples, ?).
    Parameters:
      x: (n_samples, ?) the first vector. 
        In this assignment, this will be the noise vector of shape (n_samples, z_dim), 
        but you shouldn't need to know the second dimension's size.
      y: (n_samples, ?) the second vector.
        Once again, in this assignment this will be the one-hot class vector 
        with the shape (n_samples, n_classes), but you shouldn't assume this in your code.
    '''
    # Note: Make sure this function outputs a float no matter what inputs it receives
    #### START CODE HERE ####
    combined = torch.concat((x, y), dim=1).to(torch.float32)
    #### END CODE HERE ####
    return combined


# UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_input_dimensions
def get_input_dimensions(z_dim, mnist_shape, n_classes):
    '''
    Function for getting the size of the conditional input dimensions 
    from z_dim, the image shape, and number of classes.
    Parameters:
        z_dim: the dimension of the noise vector, a scalar
        mnist_shape: the shape of each MNIST image as (C, W, H), which is (1, 28, 28)
        n_classes: the total number of classes in the dataset, an integer scalar
                (10 for MNIST)
    Returns: 
        generator_input_dim: the input dimensionality of the conditional generator, 
                          which takes the noise and class vectors
        discriminator_im_chan: the number of input channels to the discriminator
                            (e.g. C x 28 x 28 for MNIST)
    '''
    #### START CODE HERE ####
    generator_input_dim = z_dim + n_classes
    discriminator_im_chan = mnist_shape[0] + n_classes
    #### END CODE HERE ####
    return generator_input_dim, discriminator_im_chan

def test_input_dims():
    gen_dim, disc_dim = get_input_dimensions(23, (12, 23, 52), 9)
    assert gen_dim == 32
    assert disc_dim == 21

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)