# -*- coding:utf-8 -*-
"""
    @describe:run this script for training and predicting
    @author:SevenHsu
    @date:2019-03-08
"""
from resnet import ResNet
from data_loader import DataLoader


def train():
    # initialize model
    resnet = ResNet()
    # initialize DataLoader
    data_loader = DataLoader()
    # load train data
    data_loader.load_data('train')
    # load valid data
    data_loader.load_data('valid')
    resnet.fit(data_loader.train_data, data_loader.valid_data)


if __name__ == '__main__':
    train()
