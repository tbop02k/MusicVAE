import os

import torch
import torch.nn as nn
import torch.optim as optim

import preprocess
import dataloader
import musicVAE_model
import utils

import config

def save_checkpoint(
    path,
    model, 
    optimizer,     
    valid_acc,        
    optimizer_scheduler, # optinal
    ):

    save_dict = {}
    save_dict.update({'model_state_dict': model.state_dict()})
    save_dict.update({'optimizer_state_dict': optimizer.state_dict()})
    save_dict.update({'valid_acc': valid_acc})

    if optimizer_scheduler is not None:        
        save_dict.update({'optimizer_scheduler_state_dict' : optimizer_scheduler.state_dict()})
    
    torch.save(save_dict, path)
    return True

def load_checkpoint(path = config.path_model_trained):
    if os.path.isfile(path):
        checkpoint = torch.load(path)
        return checkpoint
    else:        
        raise Exception('doest not exist checkpiont file')
    
def is_save_checkpoint(
    path,
    model, 
    optimizer,   
    current_valid_acc,     
    optimizer_scheduler=None):
    
    checkpoint = load_checkpoint(path)
    if current_valid_acc > checkpoint['valid_acc']:
        save_checkpoint(path, model, optimizer, current_valid_acc, optimizer_scheduler)

    return True

if __name__ == '__main__':

    device = torch.device("cuda")
    print('current device',torch.cuda.get_device_name(0))

    train_epochs = 3000
    train_set_loader, valid_set_loader = dataloader.main()
    
    model = musicVAE_model.Model()
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer_scheduler = None
    
    # continual training from check point
    if config.is_training_from_checkpoint:        
        checkpoint = load_checkpoint()          
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'optimizer_scheduler_state_dict' in checkpoint.keys():
            optimizer_scheduler.load_state_dict(checkpoint['optimizer_scheduler_state_dict'])            

        print('start training from trained weight and optimizer state')
        
    else:                
        print('start training from zero weight')


    for epoch in range(train_epochs):
        train_loss, train_acc, valid_loss, valid_acc = 0, 0, 0, 0
        
        ## Train
        model.train()    
        for batch_idx, train_set in enumerate(train_set_loader):
            train_set = train_set.to(device)

            optimizer.zero_grad()        
            output, mu, std = model(train_set)
            
            prob = nn.Softmax(dim=2)(output)
            label = torch.argmax(prob,2)

            # loss
            
            if config.is_training_from_checkpoint:
                beta = 0.2
            else:
                beta = utils.kl_annealing(epoch, 0, 0.2)            
            loss = utils.vae_loss(prob, train_set, mu, std, beta)

            # backward
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += utils.accuracy(train_set, label).item()        
                
        train_loss = train_loss / (batch_idx + 1)
        train_acc = train_acc / (batch_idx + 1)
        
        ## Validation
        model.eval() # turn off useless layer components of Model for inference
        with torch.no_grad(): # disable gradient calculation
            for batch_idx, valid_set in enumerate(valid_set_loader):
                valid_set = valid_set.to(device)
                
                output, mu, std = model(valid_set)

                prob = nn.Softmax(dim=2)(output)
                label = torch.argmax(prob,2)

                loss = utils.vae_loss(prob, valid_set, mu, std)
                valid_loss += loss.item()
                valid_acc += utils.accuracy(valid_set, label).item()  

            valid_loss = valid_loss / (batch_idx + 1)
            valid_acc = valid_acc / (batch_idx + 1)
        
        print(f"""
        train loss : {train_loss}, train_acc : {train_acc}, valid loss : {valid_loss}, valid_acc: {valid_acc}
        """)

        if epoch % 10 ==0:
            '''
            Save model & optimizer states only 
            when current validation accuracy is better than previously saved valid accuracy
            '''
            is_save_checkpoint(
                path = config.path_model_trained, 
                model=model, 
                optimizer=optimizer, 
                current_valid_acc=valid_acc, 
                optimizer_scheduler= optimizer_scheduler)