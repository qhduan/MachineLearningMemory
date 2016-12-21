
# coding: utf-8

# In[1]:

import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score
import tensorflow as tf
from tqdm import tqdm


# In[2]:

from data import get_data


# In[3]:

X_train, y_train, X_test, y_test = get_data(one_hot=True)


# In[4]:

# 标准化
X_train = X_train / 255.0
X_test = X_test / 255.0


# In[5]:

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# In[6]:

train_len, width, height, channel = X_train.shape
test_len = X_test.shape[0]
print(train_len, width, height, channel, test_len)


# In[7]:

# 学习率
learning_rate = 0.01
# 迭代次数（批次）
n_epoch = 20
# 批次大小
batch_size = 32


# In[8]:

# 输入大小
input_size = (width, height, channel)
# 输出大小
target_size = 2


# In[9]:

tf.set_random_seed(0)


# In[10]:

# 输入占位符
X = tf.placeholder(tf.float32, [batch_size, width, height, channel])
# 输出占位符
y = tf.placeholder(tf.float32, [batch_size, target_size])


# In[11]:

stddev = 1e-1

pitch_1 = tf.Variable(tf.random_normal([5, 5, 3, 32], stddev=stddev), name='pitch_1')
pitch_1_bias = tf.Variable(tf.random_normal([32], stddev=stddev), name='pitch_1_bias')

pitch_2 = tf.Variable(tf.random_normal([3, 3, 32, 32], stddev=stddev), name='pitch_2')
pitch_2_bias = tf.Variable(tf.random_normal([32], stddev=stddev), name='pitch_2_bias')


# In[12]:

conv_1 = tf.nn.relu(
    tf.nn.bias_add(
        tf.nn.conv2d(
            X, pitch_1, strides=[1, 2, 2, 1], padding='VALID'
        ),
        pitch_1_bias,
        name='bias_add_1'
    ),
    name='relu_1'
)


# In[13]:

print(conv_1.get_shape())


# In[14]:

maxpool_1 = tf.nn.max_pool(
    conv_1,
    ksize=[1, 2, 2, 1],
    strides=[1, 2, 2, 1],
    padding='VALID',
    name='max_pool_1'
)


# In[15]:

print(maxpool_1.get_shape())


# In[16]:

conv_2 = tf.nn.relu(
    tf.nn.bias_add(
        tf.nn.conv2d(
            maxpool_1, pitch_2, strides=[1, 2, 2, 1], padding='VALID'
        ),
        pitch_2_bias,
        name='bias_add_2'
    ),
    name='relu_2'
)


# In[17]:

print(conv_2.get_shape())


# In[18]:

maxpool_2 = tf.nn.max_pool(
    conv_2,
    ksize=[1, 2, 2, 1],
    strides=[1, 2, 2, 1],
    padding='VALID',
    name='max_pool_2'
)


# In[19]:

print(maxpool_2.get_shape())


# In[20]:

flatten = tf.reshape(maxpool_2, [batch_size, -1])


# In[21]:

print(flatten.get_shape())
dim = int(flatten.get_shape()[1])
print('dim is {}'.format(dim))


# In[22]:

weight_1 = tf.Variable(tf.random_normal([dim, 512]), name='weight_1')
bias_1 = tf.Variable(tf.random_normal([512]), name='bias_1')

weight_2 = tf.Variable(tf.random_normal([512, target_size]), name='weight_2')
bias_2 = tf.Variable(tf.random_normal([target_size]), name='bias_2')


# In[23]:

full_connect_1 = tf.nn.relu(
    tf.add(
        tf.matmul(flatten, weight_1, name='matmul_1'),
        bias_1,
        name='add_1'
    ),
    name='relu_5'
)


# In[24]:

print(full_connect_1.get_shape())


# In[25]:

full_connect_2 = tf.add(
    tf.matmul(full_connect_1, weight_2, name='matmul_2'),
    bias_2,
    name='add_2'
)


# In[26]:

print(full_connect_2.get_shape())


# In[27]:

pred = full_connect_2


# In[28]:

cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(
        pred, y
    )
)


# In[29]:

opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# In[30]:

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[31]:

init = tf.global_variables_initializer()


# In[32]:

def batch_flow(inputs, targets, batch_size):
    """流动数据流"""
    flowed = 0
    total = len(inputs)
    while True:
        X_ret = []
        y_ret = []
        for i in range(total):
            X_ret.append(inputs[i])
            y_ret.append(targets[i])
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


# In[ ]:

for batch_x, batch_y in batch_flow(X_train, y_train, batch_size):
    print(batch_x.shape, batch_y.shape)
    break


# In[ ]:

with tf.Session() as sess:
    sess.run(init)
    total = None
    for epoch in range(n_epoch):
        costs = []
        accs = []
        for batch_x, batch_y in tqdm(batch_flow(X_train, y_train, batch_size), total=total):
            _, c, acc = sess.run([opt, cost, accuracy], feed_dict={X: batch_x, y: batch_y})
            costs.append(c)
            accs.append(acc)
        print('epoch: {}, loss: {:.4f}, acc: {:.4f}'.format(epoch, np.mean(costs), np.mean(accs)))
        if total is None:
            total = len(costs)
    print('calculate test accuracy')
    costs = []
    accs = []
    for batch_x, batch_y in tqdm(batch_flow(X_test, y_test, batch_size)):
        c, acc = sess.run([cost, accuracy], feed_dict={X: batch_x, y: batch_y})
        costs.append(c)
        accs.append(acc)
    print('test loss: {:.4f}, acc: {:.4f}'.format(np.mean(costs), np.mean(accs)))
    print('Done')


# 过拟合的很严重～～恩～～
