# -*- coding:utf-8 -*-
"""
    @describe:ResNet model training configuration
    @author:SevenHsu
    @date:2019-03-08
"""
# ======configuration for data loading======
data_path = "./data/cifar-10"
valid_size = 1000  # valid data size should less than 10000
# ==========================================


# ===configuration for model and training===
depth = 18  # set depth in [18,34,50,101]
height = 32  # height of input images
width = 32  # width of inout images
channel = 3  # channel of input images
num_classes = 10  # numbers of predict classes
learning_rate = 0.001  # initial learning rate
learning_decay_rate = 0.95  # decay rate of learning rate
learning_decay_steps = 1000  # decay steps of learning rate
epoch = 50  # training epochs
batch_size = 32  # batch size
model_path = "./models/"  # directory path for saving trained models
summary_path = "./summary/"  # directory path for saving training logs
# ===========================================
