import numpy as np
from scipy import stats
from tqdm import tqdm
import pandas as pd
import time
from numba import jit

FEATURES_NUM = 136


def read_dataset(path, howlines=1000, isall=False):
    if isall:
        howlines = 1e50

    k = 0
    X_train = []  # <feature-value>[46]
    y_train = []  # <label>
    Query = []  # <query-id><document-id><inc><prob>
    Query_target = set()

    with open(path, 'r') as file:
        for line in tqdm(file):
            if k == 0:
                k += 1
                continue
            k += 1
            # print(line)
            if k > howlines:
                break
            split = line.split(',')
            y_train_cur = float(split[0])
            query_cur = int(split[-1].strip())
            y_train_cur_ = y_train_cur
            # if (query_cur, y_train_cur) in Query_target:
            #     # y_train_cur_ = y_train_cur - np.random.choice(10,1)[0]
            #     continue

            y_train.append(y_train_cur_)
            X_train.append(extract_features(split))
            Query.append(query_cur)
            Query_target.add((query_cur, y_train_cur))
            if k % 10000 == 0:
                print('Read %d lines from file...' % (len(X_train)))
        return X_train, y_train, Query


def extract_features(split):
    '''
    Extract the query to document features used
    as input to the neural network
    '''
    features = []
    for i in range(1, FEATURES_NUM + 1):
        # print(float(split[i]))
        features.append(float(split[i]))
    return features


def make_sumbmit(y_pred, Query_test):
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
    sub_time = time.time()
    print("The time of your submission is {}".format(sub_time))
    sample.to_csv("data/my_sub_{}.txt".format(sub_time), index=False)

@jit
def prepare_pairs(X, y):
    print("Starting prepare your pairs ...")


    y_seen = list()
    x_pair = np.zeros((X.shape[1], 2))
    y_pair = np.zeros((2,))
    X_all = list()
    y_all = list()

    for i in tqdm(range(X.shape[0])):
        for j in range(X.shape[0]):

            if i == j or [i,j] in y_seen or [j,i] in y_seen:
                continue
            else:
                if y[i] > y[j]:
                    y_seen.append([i,j])
                    y_pair[:] = np.array([y[i], y[j]])
                    x_pair[:] = np.array([X[i], X[j]]).T
                else:
                    y_seen.append([j, i])
                    y_pair[:] = np.array([y[j], y[i]])
                    x_pair[:] = np.array([X[j], X[i]]).T

                y_all.append(y_pair)
                X_all.append(x_pair)
    return np.array(X_all), np.array(y_all)


def prepare_pairs2(data):
    all_queries = set(data.QID)
    X_pairs = dict()
    y_pairs = dict()

    features_len = len(data.columns) - 2 #two for Q and Y

    x_pair = np.zeros((features_len, 2))
    y_pair = np.zeros((2,))

    y_seen  = dict()

    for query in tqdm(all_queries):
        y_seen[query] = list()
        X_pairs[query] = list()
        y_pairs[query] = list()

        temp_data = data[data.QID == query]

        selected_columns = [i for i in range(len(data.columns) - 1) if i != 0]

        X = temp_data.iloc[:, selected_columns].values
        y = temp_data.Y.values

        for i in range(X.shape[0]):
            for j in range(X.shape[0]):

                if i == j or [i, j] in y_seen[query] or \
                        [j, i] in y_seen[query]:
                    continue
                else:
                    if y[i] > y[j]:
                        y_seen[query].append([i, j])
                        y_pair[:] = np.array([y[i], y[j]])
                        x_pair[:] = np.array([X[i], X[j]]).T
                    else:
                        y_seen[query].append([j, i])
                        y_pair[:] = np.array([y[j], y[i]])
                        x_pair[:] = np.array([X[j], X[i]]).T

                    y_pairs[query].append(y_pair)
                    X_pairs[query].append(x_pair)

    return X_pairs, y_pairs
