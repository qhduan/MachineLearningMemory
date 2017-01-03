
import numpy as np
import tensorflow as tf

def train_softmax(pred, y, opt, clip=None):
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            pred, y
        )
    )
    if clip is None:
        return opt.minimize(cost), cost
    else:
        params = tf.trainable_variables()
        gradients = tf.gradients(cost, params)
        clipped_gradients, norm = tf.clip_by_global_norm(
            gradients,
            5.0
        )
        return opt.apply_gradients(zip(clipped_gradients, params)), cost

def dropout(input_layer, dropout):
    return tf.nn.dropout(input_layer, dropout)

def true_positive(pred, y):
    return tf.count_nonzero(pred * y)

def true_negative(pred, y):
    return tf.count_nonzero((pred - 1) * (y - 1))

def false_positive(pred, y):
    return tf.count_nonzero(pred * (y - 1))

def false_negative(pred, y):
    return tf.count_nonzero((pred - 1) * y)

def accuracy(pred, y, softmax=False):
    if softmax:
        pred = tf.argmax(pred, 1)
        y = tf.argmax(y, 1)
    tp = true_positive(pred, y)
    tn = true_negative(pred, y)
    fp = false_positive(pred, y)
    fn = false_negative(pred, y)
    return (tp + tn) / (tp + fp + fn + tn)

def precision(pred, y, softmax=False):
    if softmax:
        pred = tf.argmax(pred, 1)
        y = tf.argmax(y, 1)
    tp = true_positive(pred, y)
    fp = false_positive(pred, y)
    return tp / (tp + fp)

def recall(pred, y, softmax=False):
    if softmax:
        pred = tf.argmax(pred, 1)
        y = tf.argmax(y, 1)
    tp = true_positive(pred, y)
    fn = false_negative(pred, y)
    return tp / (tp + fn)

def f1(pred, y, softmax=False):
    prec = precision(pred, y, softmax)
    reca = recall(pred, y, softmax)
    return (prec * reca * 2) / (prec + reca)

def batch_flow(inputs, targets, batch_size):
    """流动数据流"""
    flowed = 0
    total = len(inputs)
    while True:
        X_ret = []
        y_ret = []
        for i in range(total):
            X_ret.append(inputs[i])
            y_ret.append([targets[i]])
            if len(X_ret) == batch_size:
                flowed += batch_size
                X, y = np.array(X_ret), np.array(y_ret)
                y = y.reshape([batch_size, -1])
                yield X, y
                X_ret = []
                y_ret = []
            if flowed >= total:
                break
        if flowed >= total:
            break

def flatten(input_layer, name):
    f = tf.reshape(input_layer, [int(input_layer.get_shape()[0]), -1])
    print('===>', f.get_shape(), '\t', name)
    return f

def conv(input_layer, output_size, pitch_shape,
         name, strides=[1, 1, 1, 1],
         padding='VALID', activation='linear',
         batch_normalization=False):
    """卷积层"""
    with tf.variable_scope(name):
        shape = [
            pitch_shape[0],
            pitch_shape[1],
            int(input_layer.get_shape()[-1]),
            output_size
        ]
        kernel = tf.Variable(tf.random_normal(
            shape,
            stddev=np.sqrt(2.0 / (shape[0] + shape[1] + shape[3]))
        ))
        conv = tf.nn.conv2d(
            input_layer, kernel, strides=strides, padding=padding
        )

        if not batch_normalization:
            bias = tf.Variable(tf.zeros([shape[-1]]))
            conv = tf.nn.bias_add(
                conv,
                bias
            )
        else:
            beta = tf.Variable(tf.zeros([shape[-1]]), name='beta')
            gamma = tf.Variable(tf.ones([shape[-1]]), name='gamma')
            # 计算mean和variance
            batch_mean, batch_var = tf.nn.moments(
                conv, [0, 1, 2], name='moments'
            )
            conv = tf.nn.batch_normalization(
                conv, batch_mean, batch_var,
                beta, gamma, 1e-3
            )

        print('===>', conv.get_shape(), '\t', name)
        
        if activation == 'sigmoid':
            return tf.sigmoid(conv)
        elif activation == 'tanh':
            return tf.tanh(conv)
        elif activation == 'relu':
            return tf.nn.relu(conv)
        elif activation == 'softmax':
            return tf.nn.softmax(conv)
        elif activation == 'linear':
            return conv
        else:
            raise Exception('Invalid Activation')

def max_pool(input_layer, name,
             ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
             padding='VALID'):
    """最大池化"""
    with tf.variable_scope(name):
        mp = tf.nn.max_pool(input_layer, ksize=ksize, strides=strides, padding=padding)
        print('===>', mp.get_shape(), '\t', name)
        return mp

def full_connect(input_layer, output_size, name, activation='linear',
                 batch_normalization=False):
    """全连接层"""
    with tf.variable_scope(name):
        shape = [int(input_layer.get_shape()[1]), output_size]
        weight = tf.Variable(
            tf.random_normal(shape, stddev=np.sqrt(2.0 / (shape[0] + shape[1]))),
            name='weight'
        )
        if not batch_normalization:
            bias = tf.Variable(tf.zeros([shape[-1]]), name='bias')
            fc = tf.nn.bias_add(
                tf.matmul(input_layer, weight),
                bias,
                name='bias_add'
            )
        else:
            beta = tf.Variable(tf.zeros([shape[-1]]), name='beta')
            gamma = tf.Variable(tf.ones([shape[-1]]), name='gamma')
            z = tf.matmul(input_layer, weight, name='matmul')
            # 计算mean和variance
            batch_mean, batch_var = tf.nn.moments(z, [0], name='moments')
            fc = tf.nn.batch_normalization(
                z, batch_mean, batch_var,
                beta, gamma, 1e-3
            )

        print('===>', fc.get_shape(), '\t', name)

        if activation == 'sigmoid':
            return tf.sigmoid(fc)
        elif activation == 'tanh':
            return tf.tanh(fc)
        elif activation == 'relu':
            return tf.nn.relu(fc)
        elif activation == 'softmax':
            return tf.nn.softmax(fc)
        elif activation == 'linear':
            return fc
        else:
            raise Exception('Invalid Activation')
