from trax import layers as tl

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