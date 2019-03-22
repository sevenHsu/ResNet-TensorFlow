# -*- coding:utf-8 -*-
"""
    @describe:run this script for training and predicting
    @author:SevenHsu
    @date:2019-03-08
"""
import config
import argparse
import numpy as np
from PIL import Image
from resnet import ResNet
from data_loader import DataLoader
from evaluate import accuracy, f1_score


def train():
    """
    train resnet and save trained model
    :return:
    """
    # initialize model
    resnet = ResNet()
    # initialize DataLoader
    data_loader = DataLoader()
    # load train data
    data_loader.load_data('train')
    # load valid data
    data_loader.load_data('valid')
    resnet.fit(data_loader.train_data, data_loader.valid_data)


def test():
    """
    test accuracy of models
    :return:
    """
    # initialize model
    resnet = ResNet()
    # initialize DataLoader
    data_loader = DataLoader()
    # load test data
    data_loader.load_data('test')
    acc = resnet.test(data_loader.test_data)
    test_data_size = len(data_loader.test_data['labels'])
    msg = "test data size:%s | test accuracy:%s" % (test_data_size, acc)
    print(msg)


def predict(img_path):
    """
    predict image provided
    :param img_path: path of required image
    :return:
    """
    # open image file
    img = Image.open(img_path)
    img = img.resize((config.width, config.height))
    img = img.convert('RGB')
    img = np.array(img, dtype=np.float32).transpose([1, 0, 2])
    img = [img / 128 - 1]
    # initialize model
    resnet = ResNet()
    # initialize DataLoader
    data_loader = DataLoader()
    # load label names
    data_loader.load_data('label_names')
    prediction, probability = resnet.predict(img)
    msg = 'prediction:%s | probability:%s' % (data_loader.label_names[prediction], probability)
    print(msg)


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="ResNet training /testing /predicting")
    parse.add_argument('-op', '--operation', default='train', choices=['train', 'test', 'predict'], required=False,
                       help="operation should be in ['train', 'test', 'predict']")
    parse.add_argument('-img', '--image', default='./data/test_images/ship.png',
                       help="required if do predict operation ")
    args = parse.parse_args()
    if args.operation == 'train':
        train()
    elif args.operation == 'test':
        test()
    else:
        predict(args.image)
