import numpy
from keras.utils.data_utils import get_file
from keras.utils.np_utils import to_categorical

max_item_id = -1
train_history = {}
file = open('dataset/ra.train', 'r')
# with ml_zip.open('ml-10m/ratings.dat', 'r') as file:
for line in file:
    user_id, item_id, rating, timestamp = line.decode('utf-8').rstrip().split('::')
    if int(user_id) not in train_history:
        train_history[int(user_id)] = [int(item_id)]
    else:
        train_history[int(user_id)].append(int(item_id))

    if max_item_id < int(item_id):
        max_item_id = int(item_id)

test_history = {}
file = open('dataset/ra.test', 'r')
for line in file:
    user_id, item_id, rating, timestamp = line.decode('utf-8').rstrip().split('::')
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