import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import preprocess
import dataloader
import musicVAE_model

import config

def accuracy(y_true, y_pred):
    y_true = torch.argmax(y_true, axis=2)
    total_num = y_true.shape[0] * y_true.shape[1]
    
    return torch.sum(y_true == y_pred) / total_num

def kl_annealing(epoch, start, end, rate=0.9):
    return end + (start - end)*(rate)**epoch

def vae_loss(recon_x, x, mu, std, beta=0):
    logvar = std.pow(2).log()
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())    
    return BCE + (beta * KLD)

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


if __name__ == '__main__':

    device = torch.device("cuda")
    print('current device',torch.cuda.get_device_name(0))

    train_epochs = 3000
    train_set_loader, valid_set_loader = dataloader.main()
    
    model = musicVAE_model.Model(device=device)
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
        train_loss, train_acc, valid_loss, valid_acc, previous_valid_acc = 0, 0, 0, 0, 0
        
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
                beta = kl_annealing(epoch, 0, 0.2)            
            loss = vae_loss(prob, train_set, mu, std, beta)

            # backward
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += accuracy(train_set, label).item()        
                
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

                loss = vae_loss(prob, valid_set, mu, std)
                valid_loss += loss.item()
                valid_acc += accuracy(valid_set, label).item()  

            valid_loss = valid_loss / (batch_idx + 1)
            valid_acc = valid_acc / (batch_idx + 1)
        
        print(f"""
        train loss : {train_loss}, train_acc : {train_acc}, valid loss : {valid_loss}, valid_acc: {valid_acc}
        """)

        if epoch % 10 ==0:
            '''
            Save model & optimizer states only 
            when current validation accuracy is better than previous accuracy
            '''
            if previous_valid_acc < valid_acc:

                save_checkpoint(
                    path = config.path_model_trained, 
                    model=model, 
                    optimizer=optimizer, 
                    valid_acc=valid_acc, 
                    optimizer_scheduler= optimizer_scheduler)