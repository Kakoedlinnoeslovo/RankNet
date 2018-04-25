from utils import readDataset

import numpy as np
from scipy import stats
from keras import Sequential
from keras.layers import Conv2D, PReLU
from keras.regularizers import l2
from keras import initializers
from keras.layers import Input, Conv2D, PReLU, MaxPool2D, Dense, Flatten, Embedding, BatchNormalization
from keras.engine.topology import Layer
from keras.layers.merge import concatenate
from keras.models import Model
from keras.optimizers import SGD
from keras import backend as K
from keras.regularizers import l2
from keras import losses
import numpy as np
from keras.utils import to_categorical
from tqdm import tqdm
from keras import initializers
from keras import optimizers
from keras import losses
from sklearn.metrics import mean_squared_error as mse
import pandas as pd
import time



#simple pointwise approach
class Network:
    def __init__(self):
        pass

    @staticmethod
    def prelu(x, name='default'):
        if name == 'default':
            return PReLU(alpha_initializer=initializers.Constant(value=0.25))(x)
        else:
            return PReLU(alpha_initializer=initializers.Constant(value=0.25), name=name)(x)

    def _conv_block(self, input, out_dim, kernel, counter, weight_decay):
        x = Conv2D(out_dim, (kernel, kernel), name="conv_" + str(counter),
                   kernel_regularizer=l2(weight_decay), padding='same')(input)
        out = self.prelu(x)
        return out

    def _build_model(self, im_size1=17, im_size2 = 8, hidden_dim=128,
                     weight_decay=0.005, feature_dim=2, is_concated=True):

        input = Input((im_size1, im_size2, 1))

        x = BatchNormalization()(input)
        x = self._conv_block(input=x, out_dim=hidden_dim, kernel=3, counter=0, weight_decay=weight_decay)
        x = self._conv_block(input=x, out_dim=hidden_dim, kernel=3, counter=1, weight_decay=weight_decay)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name="max_pool1")(x)
        x = self._conv_block(input=x, out_dim=hidden_dim, kernel=3, counter=2, weight_decay=weight_decay)
        x1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name="max_pool2")(x)
        x2 = self._conv_block(input=x1, out_dim=hidden_dim * 2, kernel=4, counter=3, weight_decay=weight_decay)
        # x1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name="max_pool3")(x)
        # x2 = self._conv_block(input=x1, out_dim=hidden_dim * 2, kernel=2, counter=4, weight_decay=weight_decay)
        # x1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name="max_pool4")(x)
        # x2 = self._conv_block(input=x1, out_dim=hidden_dim * 2, kernel=1, counter=5, weight_decay=weight_decay)
        if is_concated:
            x_concated = concatenate([x1, x2], axis=3)
            x_flatten = Flatten()(x_concated)
        else:
            x_flatten = Flatten()(x2)
        x = Dense(units=feature_dim, kernel_regularizer=l2(weight_decay))(x_flatten)
        x = PReLU(name="deep_deatures")(x)

        y_out = Dense(1, activation="linear", kernel_regularizer=l2(weight_decay))(x)

        model = Model(inputs=input, outputs=y_out)
        model.summary()
        self.model = model


def main():
    X_train, y_train, Query = readDataset("data/train.data.cvs", howlines=10000, isall=False)
    X = np.array(X_train)#.astype(np.float64)
    y = np.array(y_train)
    #y = abs(np.array(y_train) - np.array(y_train).mean())/np.array(y_train).std()
    y = np.log(y) + 10
    #y = stats.zscore(y)
    #X = (X - X.mean())/X.std()
    X = stats.zscore(X)
    mu, sigma = 0, 0.1
    # creating a noise with the same dimension as the dataset (2,2)
    noise = np.random.normal(mu, sigma, [X.shape[0], X.shape[1]])
    X = X + noise
    print(X.shape)

    learning_rate = 0.01
    batch_size = 1000
    epochs = 1

    model_ = Network()
    X = X.reshape(X.shape[0], 17, 8, 1)
    model_._build_model(im_size1=X.shape[1], im_size2 = X.shape[2], hidden_dim=32)
    optimizer = optimizers.Adam(lr = learning_rate)
    model_.model.compile(optimizer = optimizer, loss = losses.mse)
    model_.model.fit(X[:-100], y[:-100], batch_size=batch_size, epochs=epochs,
                     validation_data=(X[-100:], y[-100:]))
    y_pred = model_.model.predict(X)
    print(mse(y, y_pred))



    X_test, y_test, Query_test = readDataset("data/testset.cvs", howlines=10000, isall=True)
    X = np.array(X_test)  # .astype(np.float64)
    X = X.reshape(X.shape[0], 17, 8, 1)
    # y = abs(np.array(y_test) - np.array(y_test).mean()) / np.array(y_test).std()
    # X = (X - X.mean()) / X.std()
    # X = stats.zscore(X)
    y_pred = model_.model.predict(X)

    sample = pd.read_csv("data/sample.txt")

    sorted_docsall = list()
    for query in tqdm(sorted(set(Query_test))):
        docs = np.array(list(sample[sample.QueryId == query].DocumentId))
        # print(docs[:10])
        # indexes = sample[sample.QueryId == query].index
        indexes = np.array([i for i in range(len(sample[sample.QueryId == query].index))])
        indexes_sorted = indexes[np.argsort(y_pred[indexes].ravel(), axis=0)]
        docs_sorted = docs[indexes_sorted]
        # print(docs_sorted[:10])
        sorted_docsall.extend(docs_sorted)
        # sample[sample.QueryId == query].DocumentId = docs_sorted

    sample.DocumentId = sorted_docsall
    sample.to_csv("data/my_sub_{}.txt".format(time.time()), index=False)

if __name__ == "__main__":
    main()
