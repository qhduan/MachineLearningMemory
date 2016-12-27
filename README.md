
# MEMORY is AWFUL, CODING FOREVER

# 记忆很丢脸，撸码到永远

!!!Chinese!!!

# 前提

    numpy
    sklearn >= 0.18 这个版本不达标，某些代码就要改……
    tensorflow

这些肯定是必须的

# 样例数据

数据尽量用sklearn或tensorflow里面自带的datasets（所谓toy data？）

# 代码

Some code reference to [Tensorflow Examples](https://github.com/aymericdamien/TensorFlow-Examples)

一些代码，风格，写作参数参考了[Tensorflow Examples](https://github.com/aymericdamien/TensorFlow-Examples)

### 线性回归(Linear Regression)实现

[Linear Regression (sklearn)](LinearRegression/sklearn_version.ipynb)

[Linear Regression (tensorflow)](LinearRegression/tensorflow_version.ipynb)

### 逻辑回归(分类)(Logistic Classification)实现

[Logistic Classification (sklearn)](LogisticClassification/sklearn_version.ipynb)

[Logistic Classification (tensorflow)](LogisticClassification/tensorflow_version.ipynb)

### 多层感知机(MLP, Multilayer Perceptron)

其实也就是人工神经网络（ANN, Artificial Neural Network）

现在一般我们就只会叫它神经网络（NN, Neural Network）

[MLP (sklearn)](MLP/sklearn_version.ipynb)

[MLP (tensorflow)](MLP/tensorflow_version.ipynb)

### 简单的递归神经网络(Recurrent Neural Network)

[RNN (tensorflow)](RNN/RNN.ipynb)

### 长短期记忆网(LSTM, Long-Short Term Memory) （这中文翻译有什么意义？）

[LSTM (tensorflow)](RNN/LSTM.ipynb)

### 简单的卷积神经网(Convolutional Neural Network)，用了MNIST

[CNN (tensorflow)](Convolution/CNN_tensorflow.ipynb)

### Batch Normalization 在Tensorflow的示例

[Batch Normalization (tensorflow)](MISC/batch_normalization.ipynb)

### NLP 任务

#### 文本分类

##### 判断一句诗是李白写的，还是杜甫写的

[TFIDF特征 sklearn](NLP/谁的诗/TFIDF_sklearn.ipynb)

[HASH特征 sklearn](NLP/谁的诗/HASH_sklearn.ipynb)

[HASH特征 基本 CNN tensorflow](NLP/谁的诗/CNN_tensorflow.ipynb)

[HASH特征 基本 LSTM tensorflow](NLP/谁的诗/LSTM_tensorflow.ipynb)

[HASH特征 加Dropout LSTM tensorflow](NLP/谁的诗/LSTM_dropout_tensorflow.ipynb)

[HASH特征 多层 LSTM tensorflow](NLP/谁的诗/LSTM_multilayer_tensorflow.ipynb)

[HASH特征 双向 LSTM tensorflow](NLP/谁的诗/LSTM_bidirectional_tensorflow.ipynb)

[HASH特征 注意力 LSTM tensorflow](NLP/谁的诗/LSTM_attention_tensorflow.ipynb)

[HASH特征 双向注意力加Dropout LSTM tensorflow](NLP/谁的诗/LSTM_bidirectional_attention_dropout_tensorflow.ipynb)

### 机器视觉任务

[超分辨率卷积网 SRCNN tensorflow](Vision/SuperResolution/SRCNN.ipynb)

[判断一张图片是喵喵还是汪汪 CNN tensorflow](Vision/CatDog/CNN_tensorflow.ipynb)

[判断一张图片是喵喵还是汪汪 加入梯度裁剪 CNN tensorflow](Vision/CatDog/CNN_CLIP_tensorflow.ipynb)

[判断一张图片是喵喵还是汪汪 加入梯度裁剪和批量标准化 CNN tensorflow](Vision/CatDog/CNN_BN_CLIP_tensorflow.ipynb)
