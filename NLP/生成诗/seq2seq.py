#!/usr/bin/env python3

"""
请使用tensorflow==1.2.0rc2
或者tensorflow>=1.2.0
"""

import os
import sys
import math
import pickle

import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.python.layers.core import Dense

print(tf.__version__)

# https://github.com/JayParks/tf-seq2seq/blob/master/seq2seq_model.py

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

sentences = pickle.load(open(os.path.join(CURRENT_DIR, '../谁的诗/诗句.dat'), 'rb'))

max_len = 0
data = []
sequence_length = []
for s in sentences['X']:
    q = s[:int(len(s)/2)]
    a = s[int(len(s)/2):]
    if len(q) > max_len:
        max_len = len(q)
    if len(q) > max_len:
        max_len = len(a)
    sequence_length.append(len(a))
    data.append((q, a))

data_size = len(data)
print('max_len is {}, size of data is {}'.format(max_len, data_size))

tf.set_random_seed(0)

start_tag = '<start>'
end_tag = '<end>'
unk_tag = '<unk>'

question_2index = {}
question_2word = {}

answer_2index = {}
answer_2word = {}

for index, word in enumerate([unk_tag] + sorted(list(set(''.join(x[0] for x in data))))):
    question_2index[word] = index
    question_2word[index] = word

for index, word in enumerate([start_tag, end_tag] + sorted(list(set(''.join(x[1] for x in data))))):
    answer_2index[word] = index
    answer_2word[index] = word

question_size = len(question_2index)
answer_size = len(answer_2index)

start_token = answer_2index[start_tag]
end_token = answer_2index[end_tag]
unk_token = question_2index[unk_tag]

print(start_tag, start_token, end_tag, end_token)
print('question_size is {}'.format(question_size))
print('answer_size is {}'.format(answer_size))

batch_size = 32

# 编码器输入，shape=(batch_size, max_len)
encoder_inputs = tf.placeholder(
    dtype=tf.int32,
    shape=(None, None),
    name='encoder_inputs'
)
# 编码器长度输入，shape=(batch_size,)
encoder_inputs_length = tf.placeholder(
    dtype=tf.int32,
    shape=(None,),
    name='encoder_inputs_length'
)
# 解码器输入，shape=(batch_size, max_len)
decoder_inputs = tf.placeholder(
    dtype=tf.int32,
    shape=(None, None),
    name='decoder_inputs'
)
# 解码器长度输入，shape=(batch_size,)
decoder_inputs_length = tf.placeholder(
    dtype=tf.int32,
    shape=(None,),
    name='decoder_inputs_length'
)

decoder_start_token = tf.ones(
    shape=(batch_size, 1),
    dtype=tf.int32
) * start_token
decoder_end_token = tf.ones(
    shape=(batch_size, 1),
    dtype=tf.int32
) * end_token

# 实际训练的解码器输入，实际上是 start_token + decoder_inputs
decoder_inputs_train = tf.concat([
    decoder_start_token, decoder_inputs
], axis=1)
# 实际训练的解码器目标，实际上是 decoder_inputs + end_token
decoder_targets_train = tf.concat([
    decoder_inputs, decoder_end_token
], axis=1)
# 输出解码器的权重，shape=(batch_size, max_len + 1)
decoder_inputs_weights = tf.placeholder(
    dtype=tf.float32,
    shape=(None, None),
    name='decoder_inputs_weights'
)

embedding_size = 128
sqrt3 = math.sqrt(3)
initializer = tf.random_uniform_initializer(
    -sqrt3,
    sqrt3,
    dtype=tf.float32
)

# 编码器的embedding
encoder_embeddings = tf.get_variable(
    name='encoder_embeddings',
    shape=(question_size, embedding_size),
    initializer=initializer,
    dtype=tf.float32
)
encoder_inputs_embedded = tf.nn.embedding_lookup(
    params=encoder_embeddings,
    ids=encoder_inputs
)

# 输入神经元
encoder_cell = tf.contrib.rnn.LSTMCell(256)

encoder_outputs, encoder_last_state = tf.nn.dynamic_rnn(
    cell=encoder_cell,
    inputs=encoder_inputs_embedded,
    sequence_length=encoder_inputs_length,
    dtype=tf.float32,
    time_major=False
)

# 解码器embedding
decoder_embeddings = tf.get_variable(
    name='ecoder_embeddings',
    shape=(answer_size, embedding_size),
    initializer=initializer,
    dtype=tf.float32
)

# 编码器输出投影
input_layer = Dense(
    64,
    dtype=tf.float32,
    name='input_projection'
)

decoder_inputs_embedded = input_layer(tf.nn.embedding_lookup(
    params=decoder_embeddings,
    ids=decoder_inputs_train
))

decoder_inputs_length_train = decoder_inputs_length + 1

training_helper = tf.contrib.seq2seq.TrainingHelper(
    inputs=decoder_inputs_embedded,
    sequence_length=decoder_inputs_length_train,
    time_major=False,
    name='training_helper'
)

# 解码器神经元
decoder_cell = tf.contrib.rnn.LSTMCell(256)

# 输出投影
output_layer = Dense(
    answer_size,
    name='output_projection'
)


initial_state = encoder_last_state

training_decoder = tf.contrib.seq2seq.BasicDecoder(
    cell=decoder_cell,
    helper=training_helper,
    initial_state=initial_state,
    output_layer=output_layer
)


max_decoder_length = tf.reduce_max(
    decoder_inputs_length_train
)

(
    decoder_outputs_train,
    decoder_last_state_train,
    decoder_outputs_length_decode
) = tf.contrib.seq2seq.dynamic_decode(
    decoder=training_decoder,
    output_time_major=False,
    impute_finished=True,
    maximum_iterations=max_decoder_length
)


decoder_logits_train = tf.identity(
    decoder_outputs_train.rnn_output
)

masks = tf.sequence_mask(
    lengths=decoder_inputs_length_train,
    maxlen=max_decoder_length,
    dtype=tf.float32,
    name='masks'
)

loss = tf.contrib.seq2seq.sequence_loss(
    logits=decoder_logits_train,
    targets=decoder_targets_train,
    weights=decoder_inputs_weights,
    average_across_timesteps=True,
    average_across_batch=True
)


trainable_params = tf.trainable_variables()
opt = tf.train.AdamOptimizer(
    learning_rate=0.001
)
gradients = tf.gradients(loss, trainable_params)
clip_gradients, _ = tf.clip_by_global_norm(
    gradients, 1.0
)
global_step = tf.Variable(0, trainable=False, name='global_step')
updates = opt.apply_gradients(
    zip(gradients, trainable_params),
    global_step=global_step
)


def embed_and_input_proj(inputs):
    return input_layer(
        tf.nn.embedding_lookup(decoder_embeddings, inputs)
    )

start_tokens = tf.ones([batch_size,], tf.int32) * start_token

decoding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
    start_tokens=start_tokens,
    end_token=end_token,
    embedding=embed_and_input_proj
)

inference_decoder = tf.contrib.seq2seq.BasicDecoder(
    cell=decoder_cell,
    helper=decoding_helper,
    initial_state=initial_state,
    output_layer=output_layer
)

(
    decoder_outputs_decode,
    decoder_last_state_decode,
    decoder_outputs_length_decode
) = tf.contrib.seq2seq.dynamic_decode(
    decoder=inference_decoder,
    output_time_major=False,
    # impute_finished=True,	# error occurs
    maximum_iterations=max_len
)

decoder_pred_decode = tf.expand_dims(
    decoder_outputs_decode.sample_id,
    -1
)


def transform_data(q, a, question_2index, answer_2index, max_len):
    x = [1] * max_len
    for ind, qq in enumerate(list(q)):
        if qq in question_2index:
            x[ind] = question_2index[qq]
        else:
            x[ind] = unk_token
    y = [1] * max_len
    w = [0] * (max_len + 1)
    for ind, aa in enumerate(list(a)):
        y[ind] = answer_2index[aa]
        w[ind] = 1.0
    w[ind + 1] = 1.0
    xl = len(q)
    yl = max_len
    return x, xl, y, yl, w

def batch_flow(
    data,
    question_2index, answer_2index, max_len, batch_size=4
):
    X = []
    Y = []
    XL = []
    YL = []
    W = []
    for q, a in data:
        if len(X) == batch_size:
            yield (
                np.array(X),
                np.array(XL),
                np.array(Y),
                np.array(YL),
                np.array(W)
            )
            X = []
            XL = []
            Y = []
            YL = []
            W = []

        x, xl, y, yl, w = transform_data(
            q, a, question_2index, answer_2index, max_len
        )

        X.append(x)
        XL.append(xl)
        Y.append(y)
        YL.append(yl)
        W.append(w)


for x, xl, y, yl, w in batch_flow(
    data, question_2index, answer_2index, max_len, 4
):
    print(x.shape, xl.shape, y.shape, yl.shape, w.shape)
    print('-' * 10)
    print(x)
    print('-' * 10)
    print(xl)
    print('-' * 10)
    print(y)
    print('-' * 10)
    print(yl)
    print('-' * 10)
    print(w)
    break

init = tf.global_variables_initializer()
n_epoch = 20
steps = int(data_size / batch_size) + 1

print('n_epoch', n_epoch)
print('steps', steps)


def get_result(outputs):
    print('\n'.join([
        ''.join([answer_2word[item[0]] for item in batch if item[0] != end_token])
        for batch in outputs
    ]))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epoch):
        print('epoch {}'.format(
            epoch
        ))
        costs = []

        for x, xl, y, yl, w in tqdm(batch_flow(
            data, question_2index, answer_2index, max_len, batch_size
        ), total=steps, file=sys.stdout):

            input_feed = {
                encoder_inputs: x,
                encoder_inputs_length: xl,
                decoder_inputs: y,
                decoder_inputs_length: yl,
                decoder_inputs_weights: w
            }
            output_feed = [updates, loss]
            _, c = sess.run(output_feed, input_feed)
            costs.append(c)
            if len(costs) >= steps:
                break
#         break
        print('cost: {:.4f}\n'.format(
            np.mean(costs)
        ))
    saver = tf.train.Saver(None)
    save_path = saver.save(
        sess,
        save_path='model/',
        global_step=global_step
    )
    print('model saved at %s' % save_path)

    for x, xl, y, yl, w in batch_flow(
            data, question_2index, answer_2index, max_len, batch_size
    ):
        input_feed = {
            encoder_inputs: x,
            encoder_inputs_length: xl
        }
        outputs = sess.run([decoder_pred_decode], input_feed)
        print(get_result(outputs[0]))
        break
