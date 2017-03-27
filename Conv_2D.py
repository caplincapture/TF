def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, \ 
    pool_ksize, pool_strides):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    """
    '''
    E.g. Within the defined scope, the weights and biases to be used by each of these layers 
    are generated into tf.Variable instances, with their desired shapes: weights = tf.Variable(
    tf.truncated_normal([IMAGE_PIXELS, hidden1_units], stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
    name='weights') biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')
    When, for instance, these are created under the hidden1 scope, the unique name given to the weights variable would be "hidden1/weights".
    Each variable is given initializer ops as part of their construction.
    '''
    input_channel_depth = int(x_tensor.get_shape()[3]) # depth of filter from tensor always same
    W = tf.Variable(tf.truncated_normal([*conv_ksize, input_channel_depth, \
    conv_num_outputs], dtype=tf.float32))
    # 4D tensor variable for weights passing in 2D convolution kernel
    # width, height, depth of channel, number of outputs
    B = tf.Variable(tf.zeros(conv_num_outputs), dtype = tf.float32)
    # bias to be added to convolution
    convnet = tf.nn.conv2d(x_tensor, W, [1, *conv_strides, 1] , padding = 'SAME')
    # convolution passing in tensor
    # matmul done in the conv2d op
    convnet += B
    output = tf.nn.max_pool(convnet, [1, *pool_ksize, 1] , [1, *pool_strides, 1], padding = 'SAME')
    # max pooling of the output of the convolution kerneling
    return output 


