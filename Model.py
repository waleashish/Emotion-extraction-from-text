from trax import layers as tl

"""
    Classifier which is used to classify tweets into respective emotions. The classifier makes use of trax layers for the purpose.

    Arguments:
        vocab_size : total size of the vocabulary
        output_dim : the output dimension of the dense layer
        embedding_dim : dimension of embedding layer (defaults to 256)
        mode : mode of the classification (defaults to 'train')
"""
def classifier(vocab_size, output_dim, embedding_dim = 256, mode = 'train'):
    embed_layer = tl.Embedding(vocab_size = vocab_size, d_feature = embedding_dim)
    mean_layer = tl.Mean(axis = 1)
    dense_layer = tl.Dense(n_units = output_dim)
    logsoftmax_layer = tl.LogSoftmax()
    
    model = tl.Serial(embed_layer,
                     mean_layer,
                     dense_layer,
                     logsoftmax_layer)
    
    return model