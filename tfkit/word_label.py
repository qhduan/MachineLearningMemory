
import numpy as np

class WordLabel(object):

    def __init__(self):
        self.PAD = '<PAD>'
        self.UNK = '<UNK>'
        self.word_index = {}
        self.index_word = {}

    def fit(self, X):
        from collections import Counter
        counter = Counter()
        for x in X:
            counter.update(list(x))
        word_list = [(k, v) for k, v in counter.items()]
        word_list = sorted(word_list, key=lambda x: x[1], reverse=True)
        word_list = [x[0] for x in word_list]
        word_list += [self.PAD, self.UNK]
        self.max_features = len(word_list)
        for i, w in enumerate(word_list):
            self.word_index[w] = i
            self.index_word[i] = w

    def fit_transform(self, X, max_len):
        self.fit(X)
        return self.transform(X, max_len)

    def transform(self, X, max_len, padding_direction='right'):
        import numpy as np
        if len(self.word_index) <= 0:
            raise Exception("NotFittedError: This WordLabel instance is not fitted yet. "
                            "Call 'fit' with appropriate arguments before using this method.")
        ret = []
        for x in X:
            if len(x) > max_len:
                if padding_direction == 'right':
                    x = x[:max_len]
                else:
                    x = x[-max_len:]
            r = []
            for w in x:
                if w in self.word_index:
                    r.append(self.word_index[w])
                else:
                    r.append(self.word_index[self.UNK])
            if len(r) < max_len:
                padding_size = max_len - len(r)
                if padding_direction == 'right':
                    r = r + [self.word_index[self.PAD]] * padding_size
                else:
                    r = [self.word_index[self.PAD]] * padding_size + r
            ret.append(r)
        return np.array(ret)

    def inverse_transform(self, y):
        ret = []
        for x in y:
            r = []
            for i in x:
                if i not in self.index_word:
                    raise Exception("ValueError: y contains new labels: [{}]".format(i))
                w = self.index_word[i]
                if w != self.PAD and w != self.UNK:
                    r.append(w)
            ret.append(''.join(r))
        return ret
