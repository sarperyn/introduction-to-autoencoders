import torch
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch import nn
from torch.functional import F
from torch.optim import Adam
import numpy as np
from tqdm.auto import tqdm
import umap
import umap.plot
import pandas as pd
from warnings import filterwarnings




    
######################################
######################################
######################################    
######################################
######################################
######################################

def visualize_dataset(example_data,example_targets,channel_size):
    
    if channel_size==3:
        fig = plt.figure()
        
        for i in range(6):
            plt.subplot(2,3,i+1)
            plt.tight_layout()
            plt.imshow(np.transpose(example_data[i],(1, 2, 0)))
            plt.title("Ground Truth: {}".format(example_targets[i]))
            plt.xticks([])
            plt.yticks([])
        
    if channel_size == 1:
        fig = plt.figure()
        for i in range(6):
            plt.subplot(2,3,i+1)
            plt.tight_layout()
            plt.imshow(example_data[i][0], cmap='gray')
            plt.title("Ground Truth: {}".format(example_targets[i]))
            plt.xticks([])
            plt.yticks([])
            

def visualize_results(model,data_loader):
    dataiter = iter(data_loader)
    
    images, labels = dataiter.next()
    batch_size = images.size(0) 
    channel_size = images.size(1)
    w_h = images.size(2)
    
    if next(model.encoder.parameters()).shape[1] != 3:
        images_flatten = images.view(batch_size, -1)
        output = model(images_flatten.to(device))
    else:
        output = model(images.to(device))
    
    images = images.numpy()
    output = output.view(batch_size, channel_size, w_h, w_h)
    output = output.cpu().detach().numpy()

    fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25,4))

    for images, row in zip([images, output], axes):
        for img, ax in zip(images, row):
            
            ax.imshow((np.transpose(img,(1, 2, 0))*255).astype(np.uint8),cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
                
                        
def visualize2d_interactive(data,targets,classes):
    
    hover_data = pd.DataFrame({'index':np.arange(len(data)),
                               'label':targets})
    
    hover_data['item'] = hover_data.label.map(classes)
    mapping = umap.UMAP().fit(data)
    umap.plot.output_notebook()
    p = umap.plot.interactive(mapping, labels=targets, hover_data=hover_data, point_size=2,)
    umap.plot.show(p)
    
def visualize2d(data,targets):
    
    mapping = umap.UMAP().fit(data)
    umap.plot.points(mapping,labels=targets, theme='fire');
    
    
    
def add_noise(tensor,std=0.7,mean=0.):
    
    return tensor + torch.normal(mean,std,tensor.size())

    
######################################
######################################
######################################
######################################
######################################
######################################