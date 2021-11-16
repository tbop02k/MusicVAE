import torch
import torch.nn as nn

import config

class Encoder(nn.Module):
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 latent_dim, 
                 num_layers=2):
        
        super(Encoder, self).__init__()
        
        self.last_concatenated_size = 2 * num_layers * hidden_size
        
        self.lstm = nn.LSTM(
            batch_first=True,
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            bidirectional=True
        )        
        
        self.mu = nn.Linear(self.last_concatenated_size, latent_dim)
        self.std = nn.Linear(self.last_concatenated_size, latent_dim)
        self.softplus = nn.Softplus()
        
    def forward(self, x):       
        
        x, (h, _) = self.lstm(x)
        concatenated_latent_vec = h.transpose(0, 1).reshape(-1, self.last_concatenated_size)
        
        # softplus to make log variance positive
        mu, std = self.mu(concatenated_latent_vec), self.softplus(self.std(concatenated_latent_vec)) 
        eps = torch.randn_like(mu)
        
        # output shape = (batch_size, concatenated output hidden vector)
        z = mu + (eps * std)
        
        return z, mu, std

class Conductor(nn.Module):
    """
    Two layered uni directional LSTM Conductor
    """
    
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 num_bars,
                 num_layers=2):
        
        super(Conductor, self).__init__()        
        self.conductor = nn.LSTM(batch_first=True,
                                 input_size=input_size,
                                 hidden_size=hidden_size,
                                 num_layers=num_layers,
                                 bidirectional=False)        
        self.num_bars = num_bars
        
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )        
        
    def forward(self, z):        
        # make axis =1
        z = z.unsqueeze(1) 
        # make sequences to get output U sequences
        z = z.repeat(1, self.num_bars, 1) 
        output, (_, _) = self.conductor(z)
        output = self.layers(output)            
        return output
    
    
class Decoder(nn.Module):
    """
    Two layered uni directional LSTM Conductor 
    followed by one fully connected layer with softmax for ouput
    """
    
    def __init__(self, 
                 input_size, 
                 hidden_size,                
                 num_layers,
                 num_bars,
                 num_units,                
                teacher_forcing=False):

        super(Decoder, self).__init__()     

        self.decoder = nn.LSTM(batch_first=True,
                               input_size=input_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               bidirectional=False)
        
        self.output = nn.Linear(hidden_size, hidden_size)

        self.num_bars = num_bars
        self.num_units = num_units
    
    def forward(self, embedding_C, input_x):       
        list_output = [] # for concat U output
        for bar_i in range(self.num_bars):
            input_embedding = embedding_C[:, bar_i, :].unsqueeze(1).repeat(1, self.num_units, 1)
            
            ''' being modified for teaching forcing...            
            input_x_bar = input_x[:, bar_i*self.num_units : (bar_i+1)*self.num_units, :]
            # embedding size + input_x_bar_size (teacher forcing)
            input_emb_x = torch.cat([input_embedding, input_x_bar], dim=2)            
            input_emb_x = torch.cat([input_embedding], dim=2)
            '''

            decoder_out, (_, _) = self.decoder(input_embedding)
            
            out = self.output(decoder_out)
            list_output.append(out)
        output = torch.cat(list_output, dim=1)        
        return output   

class ModelConfig:
    '''
    This configurations is only used for Model class
    '''

    # encoder hyperparameter    
    encoder_input_size =  2**config.num_class
    encoder_num_layers = 2
    encoder_hidden_size = 512        
    latent_dim = 2 * encoder_num_layers * encoder_num_layers
    
    # conductor
    conductor_hidden_size = 512
    conductor_num_layers = 2    

    # decoder
    decoder_input_size = conductor_hidden_size    
    decoder_hidden_size = 512    
    decoder_num_layers = 2
    

class Model(nn.Module):    
    '''
    combined model class of the encoder, conductor and decoder
    '''

    def __init__(self):
        super(Model, self).__init__()
        
        self.encoder = Encoder(
            input_size = ModelConfig.encoder_input_size, 
            hidden_size = ModelConfig.encoder_hidden_size, 
            latent_dim = ModelConfig.latent_dim,
            num_layers = ModelConfig.encoder_num_layers) 

        self.conductor = Conductor(
            input_size = ModelConfig.latent_dim, 
            hidden_size = ModelConfig.conductor_hidden_size,
            num_layers = ModelConfig.conductor_num_layers,
            num_bars = config.num_bars)

        self.decoder = Decoder(
            input_size = ModelConfig.decoder_input_size, 
            hidden_size = ModelConfig.decoder_hidden_size, 
            num_layers= ModelConfig.decoder_num_layers,
            num_bars = config.num_bars,
            num_units = config.num_units)
    
    def forward(self, x):
        z, mu, std= self.encoder(x)
        embedding_C = self.conductor(z)
        output = self.decoder(embedding_C, x)        

        return output, mu,std


    