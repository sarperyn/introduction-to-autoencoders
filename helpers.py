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
#################AE###################
######################################
######################################

def visualize_datasetAE(example_data,example_targets,dataset='mnist'):
    
    if dataset=='cifar':
        fig = plt.figure()
        
        for i in range(6):
            plt.subplot(2,3,i+1)
            plt.tight_layout()
            plt.imshow(np.transpose(example_data[i],(1, 2, 0)),interpolation='none')
            plt.title("Ground Truth: {}".format(example_targets[i]))
            plt.xticks([])
            plt.yticks([])
        
    else:
        fig = plt.figure()
        for i in range(12):
            plt.subplot(3,4,i+1)
            plt.tight_layout()
            plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
            plt.title("Ground Truth: {}".format(example_targets[i]))
            plt.xticks([])
            plt.yticks([])
            
            

def trainAE(model,train_dataloader,optimizer,criterion,EPOCH):
    
    with tqdm(total = len(train_dataloader) * EPOCH) as tt:
        model.train()
        for epoch in range(EPOCH):

            total_loss, batch_count = 0, 0



            for idx, (batch,_) in enumerate(train_dataloader):

                output = model(batch)
                
                batch = batch.view(batch.size(0),-1)
                
                loss = criterion(output,batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                total_loss += loss.item()
                batch_count += 1
                tt.update()

            total_loss = total_loss / batch_count
            print(f'{total_loss}')

            
def testAE(model,test_dataloader,criterion):
    
    test_loss, batch_count = 0, 0
    for idx,batch in enumerate(test_dataloader):
        
        data = batch[0]
        with torch.no_grad():
            output = model(data)
        
        loss = criterion(output,data.view(data.size(0),-1))
        
        test_loss += loss.item()
        batch_count += 1
        
    test_loss = test_loss / batch_count
    print(f'{test_loss}')  
    

    
    
def visualize_resultsAE(model,data_loader,data):
    dataiter = iter(data_loader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images, labels = dataiter.next()
    
    if data == 'fmnist':
        images_flatten = images.view(images.size(0), -1)
        output = model(images_flatten)
        images = images.numpy()
        output = output.view(32, 1, 28, 28)
    else:
        output = model(images.to(device))
        images = images.cpu().detach().numpy()
        output = output.view(32, 3, 32, 32)

        output = output.cpu().detach().numpy()

    fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25,4))

    for images, row in zip([images, output], axes):
        for img, ax in zip(images, row):
            
            ax.imshow((np.transpose(img,(1, 2, 0))*255).astype(np.uint8),cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)


            
            
def conv_trainAE(model,train_dataloader,optimizer,loss_fn,EPOCH=30):
    
    with tqdm(total = len(train_dataloader) * EPOCH) as tt:
        
        model.train()
        
        for epoch in range(EPOCH):
            
            total_loss, batch_count = 0, 0
            
            for idx,(batch,_) in enumerate(train_dataloader):
                
                output = model(batch)
                
                loss = loss_fn(output,batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                
                total_loss += loss.item()
                batch_count += 1
                tt.update()
                
            total_loss = total_loss / batch_count
            print(f'{total_loss}')

            
            
            
            
def visualize2d_interactive(data,targets,classes):
    
    hover_data = pd.DataFrame({'index':np.arange(len(data)),
                               'label':targets})
    
    hover_data['item'] = hover_data.label.map(classes)
    
    umap.plot.output_notebook()
    p = umap.plot.interactive(mapping, labels=targets, hover_data=hover_data, point_size=2)
    umap.plot.show(p)
    
    
def visualize2d(data,targets):
    
    mapping = umap.UMAP().fit(data)
    umap.plot.points(mapping,labels=targets, theme='fire');
    
    
    
######################################
################DAE###################
######################################
######################################
def add_noise(tensor,std=0.7,mean=0.):
    
    return tensor + torch.normal(mean,std,tensor.size())
    
def visualize_datasetDAE(example,example_targets):
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(example[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
        
        
def trainDAE(model,train_dataloader,optimizer,criterion,EPOCH):
    
    with tqdm(total=len(train_dataloader)*EPOCH) as tt:
        
        for epoch in range(EPOCH):

            total_loss, batch_count = 0, 0
            for idx, (batch,_) in enumerate(train_dataloader):

               ##################################
                denoised_data = add_noise(batch)
               ##################################


                output = model(denoised_data)

                loss = criterion(output,batch.view(batch.size(0),-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                total_loss += loss.item()
                batch_count += 1
                tt.update()

            total_loss = total_loss / batch_count
            print(f'{total_loss}')

def visualize_resultDAE(model,dataloader):    
    
    dataiter = iter(dataloader)
    images, labels = dataiter.next()
    
    denoised_images = add_noise(images)
    images_flatten = denoised_images.view(images.size(0), -1)
    
    output = model(images_flatten)
    images = images.numpy()
    output = output.view(32, 1, 28, 28)
    output = output.detach().numpy()

    fig, axes = plt.subplots(nrows=3, ncols=10, sharex=True, sharey=True, figsize=(25,4))

    for images, row in zip([denoised_images, images, output], axes):
        for img, ax in zip(images, row):
            ax.imshow(np.squeeze(img), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)