# Size of the embedding vectors (number of units in the embedding layer)
# embed dat shiz, embedding random uniform used as a lookup table
embed_size = 300 

with graph.as_default():
    embedding = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, inputs_)