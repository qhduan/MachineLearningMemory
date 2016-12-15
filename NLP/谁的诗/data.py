
import pickle

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import OneHotEncoder

data = pickle.load(open('诗句.dat', 'rb'))

X_train, X_test, y_train, y_test = train_test_split(
    data['X'], data['y'],
    test_size=0.2, random_state=0
)

print('训练集样本量：{}，测试集样本量：{}'.format(len(X_train), len(X_test)))

def fit_vectorizer(data, embedding_size, max_len, PAD):
    """用数据训练一个向量化器"""
    vectorizer = HashingVectorizer(n_features=embedding_size, analyzer='char', lowercase=False)
    words = [PAD]
    for sentences in X_train:
        t = list(sentences)
        if len(t) > max_len:
            t = t[:max_len]
        pad_size = max_len - len(t)
        if pad_size > 0:
            t = t + [PAD] * pad_size
        words += t
    vectorizer.fit(words)
    return vectorizer

def fit_onehot(data):
    onehot = OneHotEncoder()
    onehot.fit(data.reshape([-1, 1]))
    return onehot

def to_mat(data, vectorizer, max_len, PAD):
    """把一些句子转换为向量"""
    vecs = []
    for x in data:
        t = list(x)
        if len(t) > max_len:
            t = t[:max_len]
        pad_size = max_len - len(t)
        if pad_size > 0:
            t = t + [PAD] * pad_size
        vec = vectorizer.transform(list(t)).toarray()
        vecs.append(vec)
    return np.array(vecs)

def batch_flow(inputs, targets, batch_size, vectorizer, onehot, max_len, PAD):
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
                X = to_mat(X_ret, vectorizer, max_len, PAD)
                y = onehot.transform(y_ret).toarray()
                X, y = np.array(X), np.array(y)
                yield X, y
                X_ret = []
                y_ret = []
            if flowed >= total:
                break
        if flowed >= total:
            break

def test_batch_flow(inputs, targets, batch_size, vectorizer, onehot, max_len, PAD):
    for X_sample, y_sample in batch_flow(inputs, targets, batch_size, vectorizer, onehot, max_len, PAD):
        print(X_sample.shape, y_sample.shape)
        break