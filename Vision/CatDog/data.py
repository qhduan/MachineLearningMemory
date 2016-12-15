
import io
import re
import zipfile

from PIL import Image
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def extract_images_bytes(path='train.zip'):
    z = zipfile.ZipFile(path, 'r')
    images = []
    labels = []
    for file in z.filelist:
        m = re.match('.*(cat|dog).*', file.filename)
        if m:
            category = m.groups()[0]
            category = 0 if category == 'cat' else 1
            fp = z.open(file.filename)
            images.append(fp.read())
            labels.append(category)
    return images, labels

train_images, train_labels = None, None
test_images, test_labels = None, None

def get_images():
    global train_images, train_labels
    global test_images, test_labels
    if train_images is None:
        train_images, train_labels = extract_images_bytes()
        test_images, test_labels = extract_images_bytes('test.zip')
        train_images = [Image.open(io.BytesIO(image)) for image in train_images]
        test_images = [Image.open(io.BytesIO(image)) for image in test_images]
    return train_images, train_labels, test_images, test_labels

def get_data(one_hot=True):
    train_images, train_labels, test_images, test_labels = get_images()
    X_train = np.array([np.array(image) for image in train_images])
    X_test = np.array([np.array(image) for image in test_images])
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)
    if one_hot:
        one_hot_encoder = OneHotEncoder()
        y_train = one_hot_encoder.fit_transform(y_train.reshape([-1, 1])).toarray()
        y_test = one_hot_encoder.transform(y_test.reshape([-1, 1])).toarray()
    return X_train, y_train, X_test, y_test
