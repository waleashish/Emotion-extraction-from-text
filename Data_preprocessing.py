import re
import string

import pandas as pd 
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

"""
    This function processes the emotion tweet dataset and splits it into training and validation datasets.
    Arguments:
        filePath : path of data file
"""
def load_dataset(filePath):
    df = pd.read_csv(filePath)
    # Remove unwanted column from the dataframe
    df.pop('tweet_id')

    # load data into tweets and emotions datasets
    all_tweets = df['content'].tolist()
    all_emotions = df['sentiment'].tolist()

    # Training set
    train_x = all_tweets[:int(len(df) * 0.98)]
    train_y = all_emotions[:int(len(df) * 0.98)]
    # Evaluation set
    val_x = all_tweets[int(len(df) * 0.98):]
    val_y = all_emotions[int(len(df) * 0.98):]

    return df, all_tweets, all_emotions, train_x, train_y, val_x, val_y

"""
    This function extracts words from a sentence by processing it using tokenization, removal of stopwords, etc. Returns the words in
    the form of list.
    Arguments:
        text : sentence to be processed
"""
def process_sentence(text):
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    # remove unwanted text patterns (hyperlinks, hashtags, etc.)
    text = re.sub(r'\$\w*', '', str(text))
    # remove old style retweet text "RT"
    text = re.sub(r'^RT[\s]+', '', str(text))
    # remove hyperlinks
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', str(text))
    # remove hashtags
    # only removing the hash # sign from the word
    text = re.sub(r'#', '', str(text))
    # tokenize tweets
    text_tokens = word_tokenize(text)
    # make all word tokens lower case
    text_tokens = [token.lower() for token in text_tokens]
    # remove stopwords and punctuation
    text_clean = []
    for word in text_tokens:
        if word not in stopwords_english and word not in string.punctuation:
            text_clean.append(stemmer.stem(word))
    
    return text_clean


"""
    This function assigns unique integer to all the distinct words present in our dataset that we use to train the model.
    Arguments:
        train_x : training examples
"""
def build_vocab(train_x):
    """ The vocabulary includes some special tokens as follows:
    '__PAD__'  : padding
    '__</e>__' : end of line
    '__UNK__'  : unknown word (Not present in vocabulary)

    """
    vocab = {'__PAD__': 0, '__</e>__': 1, '__UNK__': 2}
    
    for x in train_x:
        for word in process_sentence(x):
            if word not in vocab:
                vocab[word] = len(vocab)
    
                
    return vocab

"""
    This function assigns unique integer to all the distinct emotion classes that we use to train the model.
    Arguments:
        emotions : set of emotions
"""
def build_emotion_vocab(emotions):
    emo_vocab = {}
    for e in emotions:
        if e not in emo_vocab:
            emo_vocab[e] = len(emo_vocab)
            
    return emo_vocab

"""
    This function converts a sentence to tensor representing integer values for words in the sentence.
    Arguments:
        sentence : sentence to be converted to tensor
        vocab_dict : dictionary containing integer for unique words
        unknown_token : unknown token used for words not present in the vocab dictionary
"""
def get_tensor(sentence, vocab_dict, unknown_token = '__UNK__'):
    tensor = []
    for word in process_sentence(sentence):
        if word in vocab_dict:
            tensor.append(vocab_dict[word])
        else:
            tensor.append(vocab_dict[unknown_token])
            
    return tensor