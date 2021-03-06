import numpy
from keras.utils.data_utils import get_file
from keras.utils.np_utils import to_categorical
from zipfile import ZipFile


def load_data():
    '''
    load data from MovieLens 100K Dataset
    http://grouplens.org/datasets/movielens/

    Note that this method uses ua.base and ua.test in the dataset.

    :return: train_users, train_x, test_users, test_x
    :rtype: list of int, numpy.array, list of int, numpy.array
    '''
    path = get_file('ml-100k.zip', origin='http://files.grouplens.org/datasets/movielens/ml-100k.zip')
    with ZipFile(path, 'r') as ml_zip:

        max_item_id = -1

        train_history = {}
        with ml_zip.open('ml-100k/ua.base', 'r') as file:
            for line in file:
                user_id, item_id, rating, timestamp = line.decode('utf-8').rstrip().split('\t')
                '''
                if rating is bigger than 3, which is 4 and 5, we consider it as positive samples
                if rating is smaller equal than 3, which is 3, 2 and 1, we consider it as negative samples.
                '''
                if rating > 3:
                    if int(user_id) not in train_history:
                        train_history[int(user_id)] = [int(item_id)]
                    else:
                        train_history[int(user_id)].append(int(item_id))
                if max_item_id < int(item_id):
                    max_item_id = int(item_id)

        print train_history

        test_history = {}
        with ml_zip.open('ml-100k/ua.test', 'r') as file:
            for line in file:
                user_id, item_id, rating, timestamp = line.decode('utf-8').rstrip().split('\t')
                if rating > 3:
                    if int(user_id) not in test_history:
                        test_history[int(user_id)] = [int(item_id)]
                    else:
                        test_history[int(user_id)].append(int(item_id))

    # item_id starts from 1
    max_item_id += 1
    train_users = list(train_history.keys())
    train_x = numpy.zeros((len(train_users), max_item_id), dtype=numpy.int32)

    for i, hist in enumerate(train_history.values()):
        mat = to_categorical(hist, max_item_id)
        train_x[i] = numpy.sum(mat, axis=0)

    test_users = list(test_history.keys())
    test_x = numpy.zeros((len(test_users), max_item_id), dtype=numpy.int32)

    for i, hist in enumerate(test_history.values()):
        mat = to_categorical(hist, max_item_id)
        test_x[i] = numpy.sum(mat, axis=0)

    return train_users, train_x, test_users, test_x
