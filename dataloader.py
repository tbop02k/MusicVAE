import pickle
from torch.utils.data import DataLoader, random_split, Dataset

import config

class MidiDataSet(Dataset):
    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx].astype('float32')
    

def main():

    with open(config.path_data_pickle, 'rb') as f:
        data = pickle.load(f)    
    
    train_size = int(config.train_ratio* len(data))
    valid_size = int(len(data) - train_size)
    batch_size = config.train_batch_size

    trainset, validset = random_split(data, [train_size, valid_size])

    train_loader = DataLoader(MidiDataSet(trainset), shuffle=True, batch_size=batch_size, num_workers=4)
    valid_loader = DataLoader(MidiDataSet(validset), shuffle=True, batch_size=batch_size, num_workers=4)


    return train_loader, valid_loader