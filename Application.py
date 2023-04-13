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
import trax.fastmath.numpy as np
from trax.supervised import training

import Data_preprocessing
import Data_generator
from Model import classifier

"""
    This function trains the classification model and returns the training loop.
    Arguments:
        model : the classifier
        train_tasks : training tasks needed for trax training
        eval_tasks : evaluation tasks needed for trax training
        n_steps : number of times the training loop is to be executed
        output_dir : the output directory
"""
def train_model(model, train_tasks, eval_tasks, n_steps, output_dir):
    training_loop = training.Loop(model,
                                 train_tasks,
                                 eval_tasks = [eval_tasks],
                                 output_dir = output_dir,
                                 random_seed = 31)
    
    training_loop.run(n_steps)
    return training_loop

"""
    This function predicts the emotion of a sentence.
    Arguments:
        sentence : the sentence for which emotion needs to be predicted
        eval_model : classification model
        vocab_dict : vocabulary dictionary
        emo_vocab_dict : enumerated emotions as dictionary
"""
def predict_emotion(sentence, eval_model, vocab_dict, emo_vocab_dict):
  input = np.array(Data_preprocessing.get_tensor(sentence, vocab_dict))
  input = input[None, :]
  pred_probs = eval_model(input)
  index_find = np.argmax(pred_probs)
  prediction = "No Prediction right now"

  for emotion, index in emo_vocab_dict.items():
    if index == index_find:
      prediction = emotion
          
  return prediction


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
    model = classifier(vocab_size=len(vocab), output_dim=len(emotion_enum))
    training_loop = train_model(model, train_task, eval_task, 2500, output_dir_expand)
    eval_model = training_loop.eval_model
    return vocab, emotion_enum, eval_model


if __name__=='__main__':
    vocab, emotion_enum, eval_model = main()
    while True:
       sentence = input("Enter sentence: ")
       print(predict_emotion(sentence, eval_model, vocab, emotion_enum))