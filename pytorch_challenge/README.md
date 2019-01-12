
# Developing an AI application

Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 

In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below. 

<img src='https://github.com/joaopamaral/pytorch_challenge/blob/master/assets/Flowers.png?raw=1' width=500px>

The project is broken down into multiple steps:

* Load and preprocess the image dataset
* Train the image classifier on your dataset
* Use the trained classifier to predict image content

We'll lead you through each part which you'll implement in Python.

When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.

First up is importing the packages you'll need. It's good practice to keep all the imports at the beginning of your code. As you work through this notebook and find you need to import a package, make sure to add the import up here.


```python
%matplotlib inline
import PIL
from PIL import Image

try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False

if IN_COLAB:
    # Download Pytorch - Google Colab - http://pytorch.org/
    from os.path import exists
    from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
    platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())
    cuda_output = !ldconfig -p|grep cudart.so|sed -e 's/.*\.\([0-9]*\)\.\([0-9]*\)$/cu\1\2/'
    accelerator = cuda_output[0] if exists('/dev/nvidia0') else 'cpu'
    pytorch_version = '0.4.1'
    !pip install -q http://download.pytorch.org/whl/{accelerator}/torch-{pytorch_version}-{platform}-linux_x86_64.whl torchvision
        
    # Check PIL Version (Bug Google Colab)
    print('PIL.PILLOW_VERSION', PIL.PILLOW_VERSION)
    # workaround 
    if PIL.PILLOW_VERSION != '5.3.0':
        !pip install Pillow==5.3.0
        print('Restart Runtime before run the notebook!')
        
    from google.colab import drive
    drive.mount('/content/gdrive')
```

    PIL.PILLOW_VERSION 5.3.0
    Drive already mounted at /content/gdrive; to attempt to forcibly remount, call drive.mount("/content/gdrive", force_remount=True).



```python
# Imports here
%matplotlib inline

import os.path as path
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import PIL.Image as Image

import matplotlib.pyplot as plt
from time import time, ctime
from collections import OrderedDict
from tqdm import tqdm
```


```python
# CUDA availability
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.')
else:
    print('CUDA is available!')
```

    CUDA is available!


## Load the data

Here you'll use `torchvision` to load the data ([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)). You can [download the data here](https://s3.amazonaws.com/content.udacity-data.com/courses/nd188/flower_data.zip). The dataset is split into two parts, training and validation. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. If you use a pre-trained network, you'll also need to make sure the input data is resized to 224x224 pixels as required by the networks.

The validation set is used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.

The pre-trained networks available from `torchvision` were trained on the ImageNet dataset where each color channel was normalized separately. For both sets you'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images.  These values will shift each color channel to be centered at 0 and range from -1 to 1.


```python
# download the dataset for google colab
!wget -q -nc https://raw.githubusercontent.com/nmpegetis/udacity-facebook-pytorch/master/pytorch_challenge/cat_to_name.json
!wget -q -nc https://s3.amazonaws.com/content.udacity-data.com/courses/nd188/flower_data.zip
!unzip -q -o flower_data.zip
```


```python
data_dir = 'flower_data'
train_dir = data_dir + '/train/'
valid_dir = data_dir + '/valid/'
gdrive_dir = 'gdrive/My Drive/udacity-facebook-pytorch/'
```


```python
# TODO: Define your transforms for the training and validation sets
std = np.array((0.229,0.224,0.225))
mean = np.array((0.485,0.456,0.406))

data_transforms = {
    'train': transforms.Compose([
          transforms.RandomRotation(30),
          transforms.RandomResizedCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.RandomVerticalFlip(),
          transforms.ToTensor(),
          transforms.Normalize(mean, std)
      ]),
    'valid': transforms.Compose([
          transforms.Resize(256),
          transforms.CenterCrop(224),
          transforms.ToTensor(),
          transforms.Normalize(mean, std)
      ]),
}


# TODO: Load the datasets with ImageFolder
image_datasets = {
    'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
    'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
}

# TODO: Using the image datasets and the trainforms, define the dataloaders
train_loader = DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=4)
valid_loader = DataLoader(image_datasets['valid'], batch_size=32, shuffle=True, num_workers=4)

```

### Label mapping

You'll also need to load in a mapping from category label to category name. You can find this in the file `cat_to_name.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/2/library/json.html). This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.


```python
import json

with open('./cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
print('flower classes: ' + str(len(cat_to_name)))
```

    flower classes: 102


### Visualize some sample data


```python
# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy() # convert images to numpy for display

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(30, 10))
for im_id in np.arange(30):
    new_im_id = im_id+1
    ax = fig.add_subplot(3, 10, new_im_id, xticks=[], yticks=[])
    image = ((np.transpose(images[im_id], (1, 2, 0)) * std) + mean).clip(min=0)
    plt.imshow(image)
    ax.set_title(cat_to_name[str(labels[im_id].item())])
```


![png](output_11_0.png)


# Building and training the classifier

Now that the data is ready, it's time to build and train the classifier. As usual, you should use one of the pretrained models from `torchvision.models` to get the image features. Build and train a new feed-forward classifier using those features.

We're going to leave this part up to you. If you want to talk through it with someone, chat with your fellow students! You can also ask questions on the forums or join the instructors in office hours.

Refer to [the rubric](https://review.udacity.com/#!/rubrics/1663/view) for guidance on successfully completing this section. Things you'll need to do:

* Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html) (If you need a starting point, the VGG networks work great and are straightforward to use)
* Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
* Train the classifier layers using backpropagation using the pre-trained network to get the features
* Track the loss and accuracy on the validation set to determine the best hyperparameters

We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!

When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right. Make sure to try different hyperparameters (learning rate, units in the classifier, epochs, etc) to find the best model. Save those hyperparameters to use as default values in the next part of the project.


```python
# Note that 'load_checkpoint' and 'save_chackpoint' should run first before 'Run all' 

# TODO: Save the checkpoint 
def save_checkpoint(model, best_epoch, best_acc, filename="checkpoint.pt"):

    torch.save(model,filename)
    
# TODO: Write a function that loads a checkpoint and rebuilds the model

def load_checkpoint(model, filename='checkpoint.pt'):
    if path.isfile(gdrive_dir + filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(gdrive_dir + filename)
        model = checkpoint
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model
```


```python
# TODO: Build and train your network

# build model 
class_to_idx = image_datasets['train'].class_to_idx
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = models.resnet152(pretrained=True)
#print(model)

# Freeze in order to not backpropagate
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 500),
                         nn.ReLU(),
                         nn.Dropout(0.5),
                         nn.Linear(500, len(cat_to_name)),
                         nn.LogSoftmax(dim=1))

model.to(device);

criterion = nn.CrossEntropyLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=.9)
```

    cuda



```python
def training_step(model, epoch=0):
    model.train()
    
    train_loss = 0.0

#     for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f'Training Model, Epoch {epoch}')): # to show more descriptions
    for batch_idx, (data, target) in enumerate(train_loader):
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)
        
    # calculate average losses
    train_loss = train_loss/len(train_loader.dataset)
    
    return train_loss
```


```python
def validation_step(model, epoch=0, criterion=None):
    model.eval()
    
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    correct_validation = 0.0
    total_analised = 0.0
    valid_loss = 0.0
    
#     for batch_idx, (data, target) in enumerate(tqdm(valid_loader, desc=f'Validating Model, Epoch {epoch}')): # to show more descriptions
    for batch_idx, (data, target) in enumerate(valid_loader):
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()

        output = model(data)
        _, pred = torch.max(output, 1)

        # compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        correct_validation += correct.sum()
        total_analised += correct.size
        
        loss = criterion(output, target)
        valid_loss += loss.item()*data.size(0)
        
        
    # calculate averages
    valid_loss = valid_loss/len(valid_loader.dataset)
    accuaracy_validation = correct_validation / len(valid_loader.dataset)
    
    return valid_loss, accuaracy_validation
```


```python
valid_loss_min = np.Inf # track change in validation loss

# load model checkpoint
if IN_COLAB and path.isfile(gdrive_dir + 'checkpoint_final.pt'):
    model = load_checkpoint(model,'checkpoint_final.pt')
    valid_loss, accuaracy_validation = validation_step(model)
    valid_loss_min = valid_loss
    print(f'\nCheckpoint Loaded! Validation Loss {valid_loss}, Validation Accuracy {accuaracy_validation*100:.6f}%')
    
if not path.isfile(gdrive_dir + 'checkpoint_final.pt'): 
    print(f'\nNo model checkpoint found!')

if not IN_COLAB:
    print(f'\nNot in Google Colab!')
```

    => loading checkpoint 'checkpoint_final.pt'
    
    Checkpoint Loaded! Validation Loss 0.27848764452491237, Validation Accuracy 94.009780%



```python
# number of epochs to train the model
n_epochs = 40

train_loss_per_epoch = []
valid_loss_per_epoch = []
valid_accuracy_per_epoch = []

# load model checkpoint
if IN_COLAB and path.isfile(gdrive_dir + 'checkpoint_final.pt'):
    model = load_checkpoint(model,'checkpoint_final.pt')
valid_loss, accuaracy_validation = validation_step(model)
valid_loss_min = valid_loss
print(f'\nCheckpoint Loaded! Validation Loss {valid_loss}, Validation Accuracy {accuaracy_validation*100:.6f}%')
    
# else:
for epoch in range(1, n_epochs+1):
    train_loss = training_step(model, epoch)
    valid_loss, accuaracy_validation = validation_step(model, epoch, criterion)


    train_loss_per_epoch.append(train_loss)
    valid_loss_per_epoch.append(valid_loss)
    valid_accuracy_per_epoch.append(accuaracy_validation)

    # print training/validation statistics 
    print(f'Epoch: {epoch} \tTraining Loss: {train_loss:.6f} \tValidation Loss: {valid_loss:.6f} \tAccuracy: {accuaracy_validation*100:.6f}%')

    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print(f'Validation loss decreased ({valid_loss_min:.6f} --> {valid_loss:.6f}).  Saving model ...')
        save_checkpoint(model, epoch, accuaracy_validation, 'checkpoint_final.pt')
        valid_loss_min = valid_loss
```

    
    Checkpoint Loaded! Validation Loss 4.642466894863287, Validation Accuracy 0.366748%
    Epoch: 1 	Training Loss: 4.454860 	Validation Loss: 4.207188 	Accuracy: 11.491443%
    Validation loss decreased (4.642467 --> 4.207188).  Saving model ...
    Epoch: 2 	Training Loss: 4.094844 	Validation Loss: 3.737140 	Accuracy: 27.017115%
    Validation loss decreased (4.207188 --> 3.737140).  Saving model ...
    Epoch: 3 	Training Loss: 3.661686 	Validation Loss: 3.176709 	Accuracy: 35.085575%
    Validation loss decreased (3.737140 --> 3.176709).  Saving model ...
    Epoch: 4 	Training Loss: 3.185140 	Validation Loss: 2.655908 	Accuracy: 45.232274%
    Validation loss decreased (3.176709 --> 2.655908).  Saving model ...
    Epoch: 5 	Training Loss: 2.785928 	Validation Loss: 2.186704 	Accuracy: 58.435208%
    Validation loss decreased (2.655908 --> 2.186704).  Saving model ...
    Epoch: 6 	Training Loss: 2.411905 	Validation Loss: 1.816143 	Accuracy: 66.136919%
    Validation loss decreased (2.186704 --> 1.816143).  Saving model ...
    Epoch: 7 	Training Loss: 2.132485 	Validation Loss: 1.524084 	Accuracy: 71.271394%
    Validation loss decreased (1.816143 --> 1.524084).  Saving model ...
    Epoch: 8 	Training Loss: 1.884147 	Validation Loss: 1.284266 	Accuracy: 77.261614%
    Validation loss decreased (1.524084 --> 1.284266).  Saving model ...
    Epoch: 9 	Training Loss: 1.708111 	Validation Loss: 1.109030 	Accuracy: 81.907090%
    Validation loss decreased (1.284266 --> 1.109030).  Saving model ...
    Epoch: 10 	Training Loss: 1.561433 	Validation Loss: 0.985925 	Accuracy: 83.374083%
    Validation loss decreased (1.109030 --> 0.985925).  Saving model ...
    Epoch: 11 	Training Loss: 1.469420 	Validation Loss: 0.888216 	Accuracy: 84.718826%
    Validation loss decreased (0.985925 --> 0.888216).  Saving model ...
    Epoch: 12 	Training Loss: 1.362880 	Validation Loss: 0.778012 	Accuracy: 86.063570%
    Validation loss decreased (0.888216 --> 0.778012).  Saving model ...
    Epoch: 13 	Training Loss: 1.275758 	Validation Loss: 0.724996 	Accuracy: 86.919315%
    Validation loss decreased (0.778012 --> 0.724996).  Saving model ...
    Epoch: 14 	Training Loss: 1.198820 	Validation Loss: 0.672439 	Accuracy: 87.408313%
    Validation loss decreased (0.724996 --> 0.672439).  Saving model ...
    Epoch: 15 	Training Loss: 1.153651 	Validation Loss: 0.613460 	Accuracy: 88.753056%
    Validation loss decreased (0.672439 --> 0.613460).  Saving model ...
    Epoch: 16 	Training Loss: 1.107029 	Validation Loss: 0.579999 	Accuracy: 89.853301%
    Validation loss decreased (0.613460 --> 0.579999).  Saving model ...
    Epoch: 17 	Training Loss: 1.073306 	Validation Loss: 0.535918 	Accuracy: 89.731051%
    Validation loss decreased (0.579999 --> 0.535918).  Saving model ...
    Epoch: 18 	Training Loss: 1.033827 	Validation Loss: 0.537671 	Accuracy: 89.486553%
    Epoch: 19 	Training Loss: 0.985597 	Validation Loss: 0.491118 	Accuracy: 90.709046%
    Validation loss decreased (0.535918 --> 0.491118).  Saving model ...
    Epoch: 20 	Training Loss: 0.946472 	Validation Loss: 0.448108 	Accuracy: 92.176039%
    Validation loss decreased (0.491118 --> 0.448108).  Saving model ...
    Epoch: 21 	Training Loss: 0.930764 	Validation Loss: 0.436350 	Accuracy: 91.931540%
    Validation loss decreased (0.448108 --> 0.436350).  Saving model ...
    Epoch: 22 	Training Loss: 0.882900 	Validation Loss: 0.424303 	Accuracy: 92.176039%
    Validation loss decreased (0.436350 --> 0.424303).  Saving model ...
    Epoch: 23 	Training Loss: 0.879180 	Validation Loss: 0.404389 	Accuracy: 92.787286%
    Validation loss decreased (0.424303 --> 0.404389).  Saving model ...
    Epoch: 24 	Training Loss: 0.889232 	Validation Loss: 0.397750 	Accuracy: 92.909535%
    Validation loss decreased (0.404389 --> 0.397750).  Saving model ...
    Epoch: 25 	Training Loss: 0.849306 	Validation Loss: 0.396742 	Accuracy: 92.298289%
    Validation loss decreased (0.397750 --> 0.396742).  Saving model ...
    Epoch: 26 	Training Loss: 0.812103 	Validation Loss: 0.366106 	Accuracy: 92.787286%
    Validation loss decreased (0.396742 --> 0.366106).  Saving model ...
    Epoch: 27 	Training Loss: 0.800565 	Validation Loss: 0.363922 	Accuracy: 93.398533%
    Validation loss decreased (0.366106 --> 0.363922).  Saving model ...
    Epoch: 28 	Training Loss: 0.787827 	Validation Loss: 0.353366 	Accuracy: 93.398533%
    Validation loss decreased (0.363922 --> 0.353366).  Saving model ...
    Epoch: 29 	Training Loss: 0.778724 	Validation Loss: 0.335228 	Accuracy: 93.765281%
    Validation loss decreased (0.353366 --> 0.335228).  Saving model ...
    Epoch: 30 	Training Loss: 0.759294 	Validation Loss: 0.332012 	Accuracy: 93.398533%
    Validation loss decreased (0.335228 --> 0.332012).  Saving model ...
    Epoch: 31 	Training Loss: 0.743088 	Validation Loss: 0.332175 	Accuracy: 93.031785%
    Epoch: 32 	Training Loss: 0.745180 	Validation Loss: 0.327996 	Accuracy: 92.909535%
    Validation loss decreased (0.332012 --> 0.327996).  Saving model ...
    Epoch: 33 	Training Loss: 0.735545 	Validation Loss: 0.301154 	Accuracy: 93.887531%
    Validation loss decreased (0.327996 --> 0.301154).  Saving model ...
    Epoch: 34 	Training Loss: 0.734687 	Validation Loss: 0.314265 	Accuracy: 93.765281%
    Epoch: 35 	Training Loss: 0.717621 	Validation Loss: 0.303470 	Accuracy: 94.009780%
    Epoch: 36 	Training Loss: 0.693645 	Validation Loss: 0.307032 	Accuracy: 93.765281%
    Epoch: 37 	Training Loss: 0.658125 	Validation Loss: 0.288625 	Accuracy: 93.887531%
    Validation loss decreased (0.301154 --> 0.288625).  Saving model ...
    Epoch: 38 	Training Loss: 0.675738 	Validation Loss: 0.291471 	Accuracy: 93.398533%
    Epoch: 39 	Training Loss: 0.661545 	Validation Loss: 0.278488 	Accuracy: 94.009780%
    Validation loss decreased (0.288625 --> 0.278488).  Saving model ...
    Epoch: 40 	Training Loss: 0.651432 	Validation Loss: 0.280541 	Accuracy: 93.765281%



```python
fig, ax1 = plt.subplots()
ax1.plot(range(1, len(train_loss_per_epoch) + 1), train_loss_per_epoch);
ax1.plot(range(1, len(valid_loss_per_epoch) + 1), valid_loss_per_epoch);
plt.legend(['Training Loss', 'Validation Loss']);
plt.xlabel('epoch');
plt.ylabel('loss');

ax2 = ax1.twinx()
ax2.plot(range(1, len(valid_accuracy_per_epoch) + 1), np.asarray(valid_accuracy_per_epoch)*100, color='r');
plt.ylabel('%');
plt.savefig('checkpoint_final.png')
```


![png](output_19_0.png)



```python
if IN_COLAB:
    !cp checkpoint_final.pt 'gdrive/My Drive/udacity-facebook-pytorch/checkpoint_final.pt'
    !cp checkpoint_final.png 'gdrive/My Drive/udacity-facebook-pytorch/checkpoint_final.png'
    print('Model Saved on Drive!')
    
```

    Model Saved on Drive!


## Save the checkpoint

Now that your network is trained, save the model so you can load it later for making predictions. You probably want to save other things such as the mapping of classes to indices which you get from one of the image datasets: `image_datasets['train'].class_to_idx`. You can attach this to the model as an attribute which makes inference easier later on.

```model.class_to_idx = image_datasets['train'].class_to_idx```

Remember that you'll want to completely rebuild the model later so you can use it for inference. Make sure to include any information you need in the checkpoint. If you want to load the model and keep training, you'll want to save the number of epochs as well as the optimizer state, `optimizer.state_dict`. You'll likely want to use this trained model in the next part of the project, so best to save it now.


```python
# Note that 'load_checkpoint' and 'save_chackpoint' should run first before 'Run all' 

# TODO: Save the checkpoint 
def save_checkpoint(model, best_epoch, best_acc, filename="checkpoint.pt"):

    torch.save(model,filename)
```

## Loading the checkpoint

At this point it's good to write a function that can load a checkpoint and rebuild the model. That way you can come back to this project and keep working on it without having to retrain the network.


```python


# TODO: Write a function that loads a checkpoint and rebuilds the model

# Note that 'load_checkpoint' and 'save_chackpoint' should run first before 'Run all' 
def load_checkpoint(model, filename='checkpoint.pt'):
    if path.isfile(gdrive_dir + filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(gdrive_dir + filename)
        model = checkpoint
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model
```

# Inference for classification

Now you'll write a function to use a trained network for inference. That is, you'll pass an image into the network and predict the class of the flower in the image. Write a function called `predict` that takes an image and a model, then returns the top $K$ most likely classes along with the probabilities. It should look like 

```python
probs, classes = predict(image_path, model)
print(probs)
print(classes)
> [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
> ['70', '3', '45', '62', '55']
```

First you'll need to handle processing the input image such that it can be used in your network. 

## Image Preprocessing

You'll want to use `PIL` to load the image ([documentation](https://pillow.readthedocs.io/en/latest/reference/Image.html)). It's best to write a function that preprocesses the image so it can be used as input for the model. This function should process the images in the same manner used for training. 

First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. This can be done with the [`thumbnail`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) or [`resize`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) methods. Then you'll need to crop out the center 224x224 portion of the image.

Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1. You'll need to convert the values. It's easiest with a Numpy array, which you can get from a PIL image like so `np_image = np.array(pil_image)`.

As before, the network expects the images to be normalized in a specific way. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`. You'll want to subtract the means from each color channel, then divide by the standard deviation. 

And finally, PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array. You can reorder dimensions using [`ndarray.transpose`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html). The color channel needs to be first and retain the order of the other two dimensions.


```python
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model
    image.thumbnail((256 if min(image.size) == image.size[0] else image.size[0], 
                     256 if min(image.size) == image.size[1] else image.size[1]))
    
    width, height = image.size   # Get dimensions

    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2

    image = image.crop((left, top, right, bottom))
    
    np_image = np.array(image)/256
    np_image = np.divide(np.subtract(np_image, mean), std)
    
    np_image = np_image.transpose(2, 0, 1)
    return torch.from_numpy(np_image).float()
```

To check your work, the function below converts a PyTorch tensor and displays it in the notebook. If your `process_image` function works, running the output through this function should return the original image (except for the cropped out portions).


```python
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    if title:
        ax.set_title(title)
    
    ax.imshow(image)
    
    return ax
```


```python
# Test function
im = Image.open("flower_data/valid/6/image_08105.jpg")
imshow(process_image(im));
```


![png](output_29_0.png)


## Class Prediction

Once you can get images in the correct format, it's time to write a function for making predictions with your model. A common practice is to predict the top 5 or so (usually called top-$K$) most probable classes. You'll want to calculate the class probabilities then find the $K$ largest values.

To get the top $K$ largest values in a tensor use [`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk). This method returns both the highest `k` probabilities and the indices of those probabilities corresponding to the classes. You need to convert from these indices to the actual class labels using `class_to_idx` which hopefully you added to the model or from an `ImageFolder` you used to load the data ([see here](#Save-the-checkpoint)). Make sure to invert the dictionary so you get a mapping from index to class as well.

Again, this method should take a path to an image and a model checkpoint, then return the probabilities and classes.

```python
probs, classes = predict(image_path, model)
print(probs)
print(classes)
> [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
> ['70', '3', '45', '62', '55']
```


```python
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    im = Image.open(image_path)
    model.eval()
    model.cpu()
    output = torch.exp(model(process_image(im).unsqueeze(0)))
    probs, classes = output.topk(topk)

    return probs.detach().numpy().ravel(), classes.detach().numpy().ravel()
```


```python
predict("flower_data/valid/1/image_06739.jpg", model)
```




    (array([0.01414523, 0.01277535, 0.01258491, 0.0121449 , 0.01213492],
           dtype=float32), array([63, 62, 46, 51, 19]))



## Sanity Checking

Now that you can use a trained model for predictions, check to make sure it makes sense. Even if the validation accuracy is high, it's always good to check that there aren't obvious bugs. Use `matplotlib` to plot the probabilities for the top 5 classes as a bar graph, along with the input image. It should look like this:

<img src='https://github.com/joaopamaral/pytorch_challenge/blob/master/assets/inference_example.png?raw=1' width=300px>

You can convert from the class integer encoding to actual flower names with the `cat_to_name.json` file (should have been loaded earlier in the notebook). To show a PyTorch tensor as an image, use the `imshow` function defined above.


```python
def view_classify(img_path):
    ''' Function for viewing an image and it's predicted classes.
    '''
    im = Image.open(img_path)

    fig, (ax1, ax2) = plt.subplots(figsize=(9,9), ncols=2)
    
    imshow(process_image(im), ax1, cat_to_name[img_path.split('/')[-2]]);
    ax1.axis('off')
    
    probs, classes = predict(img_path, model)
    ax2.barh(np.arange(5), probs)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(5),20)
    ax2.set_yticklabels([cat_to_name[str(c)] for c in np.nditer(classes)])
    ax2.set_title('Class Probability')
    ax2.set_xlim(0.0, 0.03)
    ax2.invert_yaxis()

    plt.tight_layout()
```


```python
# Display an image along with the top 5 classes
view_classify("flower_data/valid/1/image_06739.jpg")
view_classify("flower_data/valid/18/image_04278.jpg")
```


![png](output_35_0.png)



![png](output_35_1.png)

