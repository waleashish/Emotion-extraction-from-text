import random as rnd
import trax
import trax.fastmath.numpy as np
from Data_preprocessing import get_tensor
from trax import layers as tl
from trax.supervised import training

def batch_generator(batch_size, data, data_emotions, vocab_dict, emo_vocab_dict, loop=True, shuffle=True):
    """
        batch_size: no of tensors to be returned
        data: set of examples
        data_emotions: set of targets
        vocab_dict: vocabulary dictionary
        emo_vocab_dict: enumerated emotions as dictionary
        loop: boolean to indicate loop
        shuffle: boolean indicate to shuffle or not
    """
    
    len_data = len(data)
    # Initialize data index
    data_index = 0
    # Get and array with the data indexes
    data_index_lines = list(range(len_data))
    
    # shuffle lines if shuffle is set to True
    if shuffle:
        rnd.shuffle(data_index_lines)
    
    stop = False
    while not stop:
        batch = []
        batch_emotions = []
        for i in range(batch_size):
            if data_index >= len_data:
                # If loop is set to False, break once we reach the end of the dataset
                if not loop:
                    stop = True;
                    break;
                # If user wants to keep re-using the data, reset the index
                data_index = 0
                if shuffle:
                    # Shuffle the index of the positive sample
                    rnd.shuffle(data_index_lines)
            
            tweet = data[data_index_lines[data_index]]
            tensor = get_tensor(tweet, vocab_dict)
            batch.append(tensor)
            batch_emotions.append(emo_vocab_dict[data_emotions[data_index_lines[data_index]]])
            data_index += 1
            
        batch_pad = []
            
        # Find the tensor with maximum length
        max_len = max([len(t) for t in batch]) 
        # Add zeros at the end of all tensors to match max length
        for t in batch:
            n_pad = max_len - len(t)
            pad_l = [0] * n_pad
            tensor_pad = t + pad_l
            batch_pad.append(tensor_pad)
            
        
        inputs = np.array(batch_pad)
        targets = np.array(batch_emotions)
        example_weights = np.array([1] * batch_size)

        
        yield inputs, targets, example_weights

def train_generator(batch_size, data, data_emotions, vocab_dict, emo_vocab_dict, loop=True, shuffle=True):
    return batch_generator(batch_size, data, data_emotions, vocab_dict, emo_vocab_dict, loop, shuffle)

def val_generator(batch_size, data, data_emotions, vocab_dict, emo_vocab_dict, loop=True, shuffle=True):
    return batch_generator(batch_size, data, data_emotions, vocab_dict, emo_vocab_dict, loop, shuffle)

def test_generator(batch_size, data, data_emotions, vocab_dict, emo_vocab_dict, loop=True, shuffle=True):
    return batch_generator(batch_size, data, data_emotions, vocab_dict, emo_vocab_dict, loop, shuffle)

def get_train_eval_tasks(batch_size, data, data_emotions, vocab_dict, emo_vocab_dict, loop=True, shuffle=True):
    train_task = training.TrainTask(
        labeled_data = train_generator(batch_size, data,
                                     data_emotions,
                                     vocab_dict,
                                     emo_vocab_dict,
                                     loop = loop,
                                     shuffle=True),
        loss_layer = tl.WeightedCategoryCrossEntropy(),
        optimizer = trax.optimizers.Adam(0.01),
        n_steps_per_checkpoint = 50,
    )

    eval_task = training.EvalTask(
        labeled_data = val_generator(batch_size, data,
                                     data_emotions,
                                     vocab_dict,
                                     emo_vocab_dict,
                                     loop = loop,
                                     shuffle=True), 
        n_eval_batches = 20,       
        metrics = [tl.WeightedCategoryCrossEntropy(), tl.WeightedCategoryAccuracy()],
    )
    
    return train_task, eval_task
