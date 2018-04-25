from utils import read_dataset, make_sumbmit, prepare_pairs

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
from utils import prepare_pairs2
from keras.backend import int_shape
from keras.layers import Lambda

#import pyximport

#pyximport.install()
#import cythonPreparepairs


#simple pointwise approach
class Network:
    def __init__(self):
        self.lambda_ = 0.1
        self.sigma_ = 0.1

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


    def pair_loss(self, y_true, y_pred):
        #here I have only those S_i > S_j
        sigma = 10
        #C = K.log(1 + K.exp(- sigma * (y_pred[:,0] - y_pred[:,1])))

        #shape = 150

        eq1 = K.log(1 + K.exp(- sigma * (y_pred[:,0] - y_pred[:,1])))
        # eq2 = K.log(1 + K.exp(- sigma * (y_pred[:,0] - y_pred[:,1]))) \
        #             + 1/2 * sigma * (y_pred[:,0] - y_pred[:,1])
        eq0 =  1/2 * sigma * (y_pred[:,0] - y_pred[:,1])
        eq2 = eq1 + eq0
        C = K.switch(K.equal(y_true[:,0] - y_true[:,1], 0), eq2, eq1)

        # for i in range(shape):
        #     for j in range(shape):
        #         if i == j:
        #             if y_true[i,0] == y_true[j,1]:
        #                 C += 1/2 * sigma * (y_pred[i,0] - y_pred[j,1])
        return K.sum(C, axis=0)

    def pair_loss2(self, y_true, y_pred):
        sigma = 0.1

        var1 = y_true
        var2 = K.transpose(var1)
        var3 = var1 - var2
        s1 = y_pred
        s2 = K.transpose(s1)
        s3 = s1 - s2

        losses = K.sum(- K.sign(var3) * sigma / (1 + K.exp(sigma * K.sign(var3) * s3)), axis=0)
        losses = K.sum(losses, axis=0)
        return losses


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
    X_train, y_train, Query = read_dataset("data/train.data.cvs",
                                           howlines=1000,
                                           isall=False)
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

    y = y.reshape(y.shape[0], 1)
    Query = np.array(Query)
    Query = Query.reshape(Query.shape[0], 1)

    data1 = pd.read_csv("data/train.data.cvs", nrows=10)

    data = pd.DataFrame(data=np.hstack((y, X, Query)),
                        columns=data1.columns)

    #X_zero = np.zeros((int(N*(N-1)/2), X.shape[1], 2))
    #y_zero = np.zeros((int(N*(N-1)/2), 2))
    #X, y = prepare_pairs2(data)

    learning_rate = 0.01
    batch_size = 1000
    epochs = 20

    model = Network()
    model._build_model()
    model.model.compile(optimizer = optimizers.Adam(), loss= model.pair_loss2)

    X = X.reshape(X.shape[0], 17, 8, 1)

    model.model.fit(X, y, batch_size=batch_size, epochs=epochs,
                     validation_data=(X[-100:], y[-100:]))

    # for query in list(X.keys()):
    #     X_numpy = np.array(X[query])
    #     # X_numpy = X_numpy[..., np.newaxis]
    #     X_numpy = X_numpy.reshape(X_numpy.shape[0], 17, 8, 2)
    #     print(X_numpy.shape)
    #     print(y[query][0:100])
    #     model.model.fit(X_numpy, np.array(y[query]))

    # X_test, y_test, Query_test = read_dataset("data/testset.cvs", howlines=100, isall=False)
    #
    # X = np.array(X_test)  # .astype(np.float64)
    # y = np.array(y_test)
    #
    # y = y.reshape(y.shape[0], 1)
    # Query = np.array(Query)
    # Query = Query.reshape(Query.shape[0], 1)
    #
    # data1 = pd.read_csv("data/testset.cvs", nrows=10)
    #
    # data = pd.DataFrame(data=np.hstack((y, X, Query)),
    #                     columns=data1.columns)
    #
    # X, y = prepare_pairs2(data)
    # X_numpy = np.array(X[list(X.keys())[0]])
    # X_numpy = X_numpy.reshape(X_numpy.shape[0], 17, 8, 2)
    # X_numpy = np.array(X[list(X.keys())[1]])
    # # X_numpy = X_numpy[..., np.newaxis]
    # X_numpy = X_numpy.reshape(X_numpy.shape[0], 17, 8, 2)
    #
    # y_predicted = model.model.predict(X_numpy)
    # print(y_predicted[0:1000])

    X_test, y_test, Query_test = read_dataset("data/testset.cvs", howlines=10000, isall=True)
    X = np.array(X_test)

    X = X.reshape(X.shape[0], 17, 8, 1)

    y_pred = model.model.predict(X)
    make_sumbmit(y_pred, Query_test)



    #X_, y_ = cythonPreparepairs.prepare_pairs(X, y)

if __name__ == "__main__":
    main()