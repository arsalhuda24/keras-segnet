# from SegNet import prep_data
import sys
import pandas as pd
import numpy as np
from skimage.io import imread


path = 'Data/'
img_w = 256
img_h = 256
n_labels = 2

n_train = 6
n_test = 3


def label_map(labels):
    label_map = np.zeros([img_h, img_w, n_labels])
    for r in range(img_h):
        for c in range(img_w):
            label_map[r, c, labels[r][c]] = 1
    return label_map

def prep_data(mode):
    assert mode in {'test', 'train'}, \
        'mode should be either \'test\' or \'train\''
    data = []
    label = []
    df = pd.read_csv(path + mode + '.csv')
    n = n_train if mode == 'train' else n_test
    for i, item in df.iterrows():
        if i >= n:
            break
        img, gt = [imread(path + item[0])], np.clip(imread(path + item[1]), 0, 1)
        data.append(img)
        label.append(label_map(gt))
        sys.stdout.write('\r')
        sys.stdout.write(mode + ": [%-20s] %d%%" % ('=' * int(20. * (i + 1) / n - 1) + '>',
                                                    int(100. * (i + 1) / n)))
        sys.stdout.flush()
    sys.stdout.write('\r')
    sys.stdout.flush()
    data, label = np.array(data), np.array(label).reshape((n, img_h * img_w, n_labels))

    # print (mode + ': OK')
    # print ('\tshapes: {}, {}'.format(data.shape, label.shape))
    # print('\ttypes:  {}, {}'.format(data.dtype, label.dtype))
    # print ('\tmemory: {}, {} MB'.format(data.nbytes / 1048576, label.nbytes / 1048576))

    return data, label


train_data,train_label= prep_data('train')

print(train_data.shape)
train_new=train_data.reshape(6,256,256,1)
print(train_new)
print(train_label.shape)