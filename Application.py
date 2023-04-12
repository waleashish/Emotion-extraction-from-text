"""
    This is the main function which we want to run at first in our program.
    Algorithm:
    Load the data into training and validation sets
    Build vocabulary, a dictionary mapping every unique word to an integer
    Convert tweet to tensor
    Create a batch generator which generates batch of inputs, comes very handy in AI modules and is a common and good practice
    Create model using trax
    Create trax training and evaluation tasks, use adam optimizer for training tasks
    Run the training and evaluation tasks with trax loop for given no. of steps
"""

import os
import trax
import random as rnd
from trax import fastmath
from trax import layers as tl


import Data_preprocessing
import Data_generator



def main():
    df, all_tweets, all_emotions, train_x, train_y, val_x, val_y = Data_preprocessing.load_dataset(filePath = "data/tweet_emotions.csv")
    vocab = Data_preprocessing.build_vocab(train_x)
    emotion_enum = Data_preprocessing.build_emotion_vocab(set(df.get('sentiment')))
    output_dir = '/content'
    output_dir_expand = os.path.expanduser(output_dir)
    train_task, eval_task = Data_generator.get_train_eval_tasks(500, all_tweets,
                                     all_emotions,
                                     vocab,
                                     emotion_enum,
                                     loop=True,
                                     shuffle=True)

    


if __name__=='__main__':
    main()
