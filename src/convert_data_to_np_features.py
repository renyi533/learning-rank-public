import os
import time
import numpy as np
import sys

QUERY_IDS = 'query_ids'
FEATURES = 'features'
LABEL_LIST = 'label_list'

def read_libsvm_data(filename):
    start_time = time.time()
    label_list = list()
    features = list()
    current_row = 0
    print('start loading data: '+filename)
    with open(filename, 'r') as f:
        for line in f:
            q2 = line.split(" ")
            label_list.append(q2[0])
            del q2[0]
            d = ':'.join(map(str, q2))
            e = d.split(":")
            features.append(e[1::2])
            if current_row % 10000 == 0:
                print('row %d - %f seconds' % (current_row, time.time() - start_time))
                print('label:'+str(label_list[current_row]))
                print('features:'+str(features[current_row]))
            current_row += 1

    print('Done loading data - %f seconds' % (time.time() - start_time))

    label_list = np.asarray(label_list, dtype=float)
    features = np.asarray(features, dtype=float)
    query_ids = np.asarray(features[:, 0], dtype=int)
    features = features[:, 1:]
    return label_list, query_ids, features

def convert(filename, target_dir):
    label_list, query_ids, features = read_libsvm_data(filename)
    np_file_directory = target_dir

    if not os.path.exists(np_file_directory):
        os.makedirs(np_file_directory)

    np.save(os.path.join(np_file_directory, LABEL_LIST), label_list)
    np.save(os.path.join(np_file_directory, FEATURES), features)
    np.save(os.path.join(np_file_directory, QUERY_IDS), query_ids)


if __name__ == '__main__':
    # converters = {
    #     0: lambda x: int(x),
    #     1: lambda x: str(x).split(':')[1], #int(str(x).split(':')[1]),
    # }
    # for i in range(2,136):
    #     converters[i] = lambda x: float(str(x).split(':')[1])

    if len(sys.argv) < 3:
        print('usage: python script.py src_file target_dir')
        sys.exit(0)

    print('Loading data...')
    filename = sys.argv[1]
    target_dir = sys.argv[2]
    # train_data = np.genfromtxt(train_data_path, delimiter=' ', converters=converters, dtype=None)
    # print('Done loading data - %f seconds' % (time.time() - start_time))


    convert(filename, target_dir)
