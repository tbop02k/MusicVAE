'''
note num to integrated note num

https://magenta.tensorflow.org/datasets/groove
'''

roland_to_paper = {
    36 : 36,
    38 : 38,
    40 : 38,
    37 : 38,
    48 : 50,
    50 : 50,
    45 : 47,
    47 : 47,
    43 : 43,
    58 : 43,
    46 : 46,
    26 : 46,
    42 : 42,
    22 : 42,
    44 : 42,
    49 : 49,
    55 : 49,
    57 : 49,
    52 : 49,
    51 : 51,
    59 : 51,
    53 : 51,
}

# make roland note to onehot index
paper_to_idx = {key : idx for idx, key in enumerate(set(roland_to_paper.values()))}
roland_to_idx = {
    key : paper_to_idx[value] for key, value in roland_to_paper.items()
}
num_class = len(set(roland_to_idx.values()))

num_bars = 4
num_units = 4 # number of quantized note per bar (1 sequence)
num_sequence = num_bars * num_units

dir_glob_midi = './groove-v1.0.0-midionly/**/*.mid'
path_data_pickle = './data.pkl'

# for training
path_model_trained = './model_trained.pth'
train_batch_size = 64
train_ratio = 0.75
is_training_from_checkpoint = True

    