import glob
import os
import pickle

import numpy as np
import pandas as pd
import pretty_midi

import config

def midi_reader(filepath):
    return pretty_midi.PrettyMIDI(filepath)    

def get_time_features(
    pm,
    num_units=config.num_units,
):
    start_time = pm.get_onsets()[0]
    end_time = pm.get_end_time()
    beats = pm.get_beats(start_time)
    beat_time = (beats[1]-beats[0]) / num_units
    
    return start_time, end_time, beat_time
        
def get_drum_inst(pm):    
    drum_inst = [i for i in pm.instruments if i.is_drum==True][0]
    return drum_inst


def make_drum_tensor(
    drum_inst,
    start_time,
    end_time,
    beat_time
):
    """
    Args:
        start_time : start time of first note
        end_time : end time of last note
        drum_inst : instance of drum_inst from pretty_midi class
    
    --------
    Return 
    
        shape: num_beats x num_class
    
        ex)
        [
            [0,1,0,1,1...], # 16 bit information
            [....],
        ]
    """
    
    num_beats = int((end_time - start_time) // beat_time) + 2
    drum_tensor = np.zeros((num_beats, config.num_class))

    for note in drum_inst.notes:    
        inst_start_ind = int(round((note.start - start_time) / beat_time))
        inst_end_ind = int(max(inst_start_ind + 1, round((note.end - start_time) / beat_time)))

        if note.pitch not in config.roland_to_idx:
            print('does not exist our pitch standard',note.pitch)
            continue
        else:
            class_idx = config.roland_to_idx[note.pitch]
        
        drum_tensor[np.arange(inst_start_ind, inst_end_ind, 1), class_idx] = 1
        
    return drum_tensor

def onehot_encoding(
    drum_tensor
):       
    """
    Args:
        drum_tensor: (np.array) beats x num_class
        
    ex)
        [1,0,0] => [1,0,0,0,0,0,0,0]
    
    --------
    Return 
        (np.array) beats x 2**num_class
    """
    
    def class_to_onehot(
        drum_tensor_bit, 
        num_class
    ):      
        onehot_class = int(np.sum([2**idx*i for idx, i in enumerate(drum_tensor_bit)]))
        drum_onehot = np.zeros((2**num_class))
        drum_onehot[onehot_class] = 1
        return drum_onehot
    
    num_class = drum_tensor.shape[1]        
    drum_tensor_onehot = [
        class_to_onehot(drum_tensor_bit, num_class) for drum_tensor_bit in drum_tensor        
    ]    
    
    return np.array(drum_tensor_onehot)

def windowing(
    drum_tensor,
    num_bars=config.num_bars,
    num_units=config.num_units,
    stride_bars=1,
):           
    
    stride_seq = stride_bars * num_units
    num_sequence_window = num_bars * num_units

    num_windows = drum_tensor.shape[0] // num_sequence_window  # ignore last rest bars

    new_list = []
    for i in range(num_windows):
        start_idx = int(stride_seq * i)
        end_idx = int(start_idx + num_sequence_window)
        new_list.append(np.expand_dims(drum_tensor[start_idx:end_idx], axis=0))
    
    if len(new_list) == 0:
        return False
    else:
        return np.vstack(new_list)

def check_4_4(pm):
    try:
        assert pm.time_signature_changes[0].numerator == 4
        assert pm.time_signature_changes[0].denominator == 4
        return True
    except:
        return False

def main(
    filepath
):
    pm = midi_reader(filepath)       

    if check_4_4(pm) is False:
        return False        

    drum_inst = get_drum_inst(pm)
    start_time, end_time, beat_time = get_time_features(pm)    
    
    drum_tensor = make_drum_tensor(drum_inst, start_time, end_time, beat_time)    
    drum_tensor = onehot_encoding(drum_tensor)        
    drum_tensor = windowing(drum_tensor)

    if drum_tensor is False:
        return False
    else:
        return drum_tensor
        

if __name__ == '__main__':

    data = []
    cnt = 0 

    for filepath in glob.glob(config.dir_glob_midi, recursive=True):
        try:
            res = main(filepath)
            if res is False:
                continue            
            else:            
                data.extend(res)
                cnt += 1

        except BaseException as e:
            print(filepath)
            print(e)            

    print('total processed midi files', cnt)

    with open(config.path_data_pickle,'wb') as f:
        pickle.dump(data, f)