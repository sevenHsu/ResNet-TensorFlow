# -*- coding:utf-8 -*-
"""
    @describe:ResNet model script contain train and predict function
    @author:SevenHsu
    @date:2019-03-08
"""
import os
import config
import tensorflow as tf
from data_loader import batch_itr
from evaluate import accuracy, f1_score


class ResNet(object):
    def __init__(self,
                 depth=config.depth,
                 height=config.height,
                 width=config.width,
                 channel=config.channel,
                 num_classes=config.num_classes,
                 learning_rate=config.learning_rate,
                 learning_decay_rate=config.learning_decay_rate,
                 learning_decay_steps=config.learning_decay_steps,
                 epoch=config.epoch,
                 batch_size=config.batch_size,
                 model_path=config.model_path,
                 summary_path=config.summary_path):
        """

        :param depth:
        """
        self.depth = depth
        self.height = height
        self.width = width
        self.channel = channel
        self.learning_rate = learning_rate
        self.learning_decay_rate = learning_decay_rate
        self.learning_decay_steps = learning_decay_steps
        self.epoch = epoch
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.model_path = model_path
        self.summary_path = summary_path
        self.num_block_dict = {18: [2, 2, 2, 2],
                               34: [3, 4, 6, 3],
                               50: [3, 4, 6, 3],
                               101: [3, 4, 23, 3]}
        self.bottleneck_dict = {18: False,
                                34: False,
                                50: True,
                                101: True}
        self.filter_out = [64, 128, 256, 512]
        self.filter_out_last_layer = [256, 512, 1024, 2048]
        self.conv_out_depth = self.filter_out[-1] if self.depth < 50 else self.filter_out_last_layer[-1]
        assert self.depth in self.num_block_dict, 'depth should be in [18,34,50,101]'
        self.num_block = self.num_block_dict[self.depth]
        self.bottleneck = self.bottleneck_dict[self.depth]
        self.input_x = tf.placeholder(tf.float32, shape=[None, self.height, self.width, self.channel], name='input_x')
        self.input_y = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='input_y')
        self.prediction = None
        self.probability = None
        self.loss = None
        self.acc = None
        self.global_step = None
        self.optimizer = None
        self.model()

    def model(self):
        """
        ResNet architecture
        :return:
        """
        # first convolution layers
        x = self.conv(x=self.input_x, k_size=3, filters_out=64, strides=2, activation=True, name='First_Conv')
        x = tf.layers.max_pooling2d(x, pool_size=[3, 3], strides=2, padding='same', name='max_pool')
        # stack blocks
        x = self.stack_block(x)
        x = tf.layers.average_pooling2d(x, pool_size=x.get_shape()[1:3], strides=1, name='average_pool')
        x = tf.reshape(x, [-1, 1 * 1 * self.conv_out_depth])
        fc_W = tf.truncated_normal_initializer(stddev=0.1)
        logits = tf.layers.dense(inputs=x, units=self.num_classes, kernel_initializer=fc_W, name='dense_layer')
        # computer prediction
        self.prediction = tf.argmax(logits, axis=-1)
        # probability
        self.probability = tf.reduce_max(tf.nn.softmax(logits), axis=-1)
        # compute accuracy
        self.acc = accuracy(logits, self.input_y)
        # loss function
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.input_y))
        # global steps
        self.global_step = tf.train.get_or_create_global_step()
        # decay learning rate
        learning_rate = tf.train.exponential_decay(learning_rate=self.learning_rate,
                                                   global_step=self.global_step,
                                                   decay_rate=self.learning_decay_rate,
                                                   decay_steps=self.learning_decay_steps,
                                                   staircase=True)
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def stack_block(self, input_x):
        """

        :param input_x:
        :return:
        """
        for stack in range(4):
            stack_strides = 1 if stack == 0 else 2
            stack_name = 'stack_%s' % stack
            for block in range(self.num_block[stack]):
                shortcut = input_x
                block_strides = stack_strides if block == 0 else 1
                block_name = stack_name + '_block_%s' % block
                filters = None
                # using bottleneck
                if self.bottleneck:
                    for layer in range(3):
                        filters = self.filter_out[stack] if layer < 2 else self.filter_out_last_layer[stack]
                        k_size = 3 if layer == 1 else 1
                        layer_strides = block_strides if layer < 1 else 1
                        activation = True if layer < 2 else False
                        layer_name = block_name + '_layer_%s' % layer
                        input_x = self.conv(x=input_x, filters_out=filters, k_size=k_size, strides=layer_strides,
                                            activation=activation, name=layer_name)
                else:
                    for layer in range(2):
                        filters = self.filter_out[stack]
                        k_size = 3
                        layer_strides = block_strides if layer < 1 else 1
                        activation = True if layer < 1 else False
                        layer_name = block_name + '_layer_%s' % layer
                        input_x = self.conv(x=input_x, filters_out=filters, k_size=k_size, strides=layer_strides,
                                            activation=activation, name=layer_name)
                # Adding shortcut and outputs of last layer
                shortcut_depth = shortcut.get_shape()[-1]
                input_x_depth = input_x.get_shape()[-1]
                with tf.name_scope(block_name + '_shortcut'):
                    if shortcut_depth != input_x_depth:
                        connect_k_size = 1
                        connect_strides = block_strides
                        connect_filter = filters
                        shortcut = self.conv(x=shortcut, filters_out=connect_filter, k_size=connect_k_size,
                                             strides=connect_strides, activation=False, name=block_name + '_shortcut')
                    input_x = tf.nn.relu(shortcut + input_x, name=block_name + '_shortcut_connect')

        return input_x

    def conv(self, x, k_size, filters_out, strides, activation, name):
        """
        convolution layer followed BN and activation(relu)layer
        :param x: input
        :param k_size: kernel_size of convolution layer
        :param filters_out: filters of convolution layer
        :param strides: strides of convolution layer
        :param activation: activate not activate with relu.[boolean]
        :param name: the name of layer
        :return: output feature map
        """
        x = tf.layers.conv2d(x, filters=filters_out, kernel_size=k_size, strides=strides, padding='same',
                             name=name + '_conv')
        x = tf.layers.batch_normalization(x, name=name + '_BN')
        if activation:
            x = tf.nn.relu(x, name=name + '_relu')
        return x

    def fit(self, train_data, valid_data):
        """
        training models
        :param train_data: contain training data and GT labels
        :param valid_data: contain validating data and GT labels
        :return: save trained models to model path
        """
        # Initialize model path and summary path
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.summary_path):
            os.makedirs(self.summary_path)

        # Initialize train_steps
        train_steps = 0
        best_valid_acc = 0.0

        # Initialize summary
        tf.summary.scalar('loss', self.loss)
        merged = tf.summary.merge_all()

        # Initialize session
        sess = tf.Session()
        writer = tf.summary.FileWriter(self.summary_path, sess.graph)
        saver = tf.train.Saver(max_to_keep=10)
        sess.run(tf.global_variables_initializer())
        # start training
        for epoch in range(self.epoch):
            for x, y in batch_itr(train_data, self.batch_size):
                train_steps += 1
                feed_dict = {self.input_x: x, self.input_y: y}
                _, train_loss, train_acc = sess.run([self.optimizer, self.loss, self.acc], feed_dict=feed_dict)
                if train_steps % 1 == 0:
                    val_loss, val_acc = sess.run([self.loss, self.acc],
                                                 feed_dict={self.input_x: valid_data['data'],
                                                            self.input_y: valid_data['labels']})
                    msg = 'epoch:%s | steps:%s | train_loss:%.4f | val_loss:%.4f | train_acc:%.4f | val_acc:%.4f' % (
                        epoch, train_steps, train_loss, val_loss, train_acc, val_acc)
                    print(msg)
                    summary = sess.run(merged,
                                       feed_dict={self.input_x: valid_data['data'], self.input_y: valid_data['labels']})
                    writer.add_summary(summary, global_step=train_steps)
                    if val_acc >= best_valid_acc:
                        best_valid_acc = val_acc
                        saver.save(sess, save_path=self.model_path, global_step=train_steps)

        sess.close()

    def test(self, test_data):
        """
        testing
        :param test_data: icontain training data and GT labels
        :return: predict value with shape[N,1],(np.ndarray)
        """
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(self.model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        feed_dict = {self.input_x: test_data['data'], self.input_y: test_data['labels']}
        _, acc = sess.run([self.prediction, self.acc], feed_dict=feed_dict)

        sess.close()
        return acc

    def predict(self, x):
        """
        predicting
        :param x: inputs images with shape[N,H,W,C],(np.ndarray)
        :return: predict value with shape[N,1],(np.ndarray)
        """
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(self.model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        prediction, probability = sess.run([self.prediction, self.probability], feed_dict={self.input_x: x})

        sess.close()
        return prediction[0], probability[0]
