# -*- coding:utf-8 -*-
"""
    @describe:load data and preprocess data
    @author:SevenHsu
    @date:2019-03-08
"""
import os
import config
import pickle
import numpy as np


class DataLoader(object):
    def __init__(self,
                 data_path=config.data_path,
                 valid_size=config.valid_size,
                 num_classes=config.num_classes):
        self.data_path = data_path
        self.valid_size = valid_size
        self.num_classes = num_classes
        self.train_data = None
        self.valid_data = None
        self.test_data = None
        self.label_names = None

    def load_data(self, set_type):
        """
        load data from bytes files
        :param set_type: 'train'/'valid'/'test'/'label_names'.[str]
        :return:
        """
        assert set_type in ['train', 'valid', 'test', 'label_names'], \
            "parameter set_type should be in ['train'/'valid'/'test']"
        # load training data
        if set_type == 'train':
            self.train_data = {'data': np.zeros(shape=[50000, 32, 32, 3], dtype=np.float32),
                               'labels': np.zeros(shape=[50000, 10], dtype=np.float32)}
            for i in range(1, 6):
                train_data_path = os.path.join(self.data_path, "data_batch_%s" % i)
                with open(train_data_path, 'rb') as fr:
                    data_batch = (pickle.load(fr, encoding='bytes'))
                # normalization
                normal_train_data = self.normalization(
                    data_batch[b'data'].reshape(10000, 3, 32, 32).transpose([0, 3, 2, 1]))
                self.train_data['data'][(i - 1) * 10000:i * 10000] = normal_train_data
                self.train_data['labels'][(i - 1) * 10000:i * 10000] = self.one_hot(data_batch[b'labels'],
                                                                                    self.num_classes)
        elif set_type in ['valid', 'test']:
            test_batch_path = os.path.join(self.data_path, "test_batch")
            with open(test_batch_path, 'rb') as fr:
                test_batch = pickle.load(fr, encoding='bytes')
            # load validation data
            if set_type == 'valid':
                self.valid_data = dict()
                normal_valid_data = self.normalization(
                    test_batch[b'data'][0:self.valid_size].reshape(self.valid_size, 3, 32, 32).transpose([0, 3, 2, 1]))
                self.valid_data['data'] = np.float32(normal_valid_data)
                self.valid_data['labels'] = self.one_hot(test_batch[b'labels'][0:self.valid_size], self.num_classes)
            # load testing data
            else:
                self.test_data = dict()
                normal_test_data = (test_batch[b'data'].reshape(10000, 3, 32, 32).transpose([0, 3, 2, 1]))
                self.test_data['data'] = np.float32(normal_test_data)
                self.test_data['labels'] = self.one_hot(test_batch[b'labels'], self.num_classes)
            del test_batch
        else:
            label_names_path = os.path.join(self.data_path, 'batches.meta')
            with open(label_names_path, 'rb') as fr:
                labels = pickle.load(fr, encoding='bytes')
                self.label_names = [i.decode('utf-8') for i in labels[b'label_names']]

    @staticmethod
    def one_hot(labels, num_class, dtype=np.float32):
        """
        transform labels to one hot matrix
        :param labels: ground truth labels list or array
        :param num_class: number of classes,width of one hot matrix
        :param dtype: data type of labels,default:float32
        :return: one hot matrix
        """
        labels_len = len(labels)
        one_hot_matrix = np.zeros(shape=[labels_len, num_class], dtype=dtype)
        for i in range(labels_len):
            one_hot_matrix[i][labels[i]] = 1
        return one_hot_matrix

    @staticmethod
    def normalization(data):
        """
        normalize data to the range[-1,1]
        :param data: data need normalize
        :return:
        """
        data = data / 128 - 1
        return data


def batch_itr(data, batch_size, shuffle=True):
    """
    split data to batches and shuffle data
    :param data: dataset for splitting to batches and shuffling.[dict]
    :param batch_size: batch size of training set
    :param shuffle: shuffle data or not,default:True.[boolean]
    :return:
    """
    data_len = len(data['data'])
    # shuffle data
    if shuffle:
        shuffle_index = np.random.permutation(data_len)
        data['data'] = data['data'][shuffle_index]
        data['labels'] = data['labels'][shuffle_index]
    # generate batch data
    for i in range(0, data_len, batch_size):
        end_index = min(i + batch_size, data_len)
        yield data['data'][i:end_index], data['labels'][i:end_index]
