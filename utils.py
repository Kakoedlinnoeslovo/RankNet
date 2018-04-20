import numpy as np
from scipy import stats
from tqdm import tqdm
import pandas as pd
import time

FEATURES_NUM = 136


def readDataset(path, howlines=1000, isall=False):
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
            X_train.append(extractFeatures(split))
            Query.append(query_cur)
            Query_target.add((query_cur, y_train_cur))
            if k % 10000 == 0:
                print('Read %d lines from file...' % (len(X_train)))
        return X_train, y_train, Query


def extractFeatures(split):
    '''
    Extract the query to document features used
    as input to the neural network
    '''
    features = []
    for i in range(1, FEATURES_NUM + 1):
        # print(float(split[i]))
        features.append(float(split[i]))
    return features