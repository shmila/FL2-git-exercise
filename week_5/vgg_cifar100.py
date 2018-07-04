import cPickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math

import os,shutil,errno

from numba.cuda.simulator import kernel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

D = 32
types = {'conv': 'conv', 'pool': 'pool', 'fc': 'fc'}


def get_image(cifar_raw_data):
    img_r = cifar_raw_data[0:1024].reshape((D, D))
    img_g = cifar_raw_data[1024:2048].reshape((D, D))
    img_b = cifar_raw_data[2048:3072].reshape((D, D))
    img = np.dstack((img_r, img_g, img_b))
    return img


def get_data():
    path_train = '/home/rocket/PycharmProjects/test/git_team/week_5/cifar-100-python/train'
    path_test = '/home/rocket/PycharmProjects/test/git_team/week_5/cifar-100-python/test'
    with open(path_train, 'rb') as tr:
        dict_train = cPickle.load(tr)
    with open(path_test, 'rb') as tes:
        dict_test = cPickle.load(tes)

    tr_data = dict_train['data']
    test_data = dict_test['data']
    n_tr_examples = tr_data.shape[0]
    n_test_examples = test_data.shape[0]
    train_x = np.zeros((n_tr_examples, D, D, 3))
    test_x = np.zeros((n_test_examples, D, D, 3))

    for i in range(n_tr_examples):
        train_x[i] = get_image(tr_data[i])
    for i in range(n_test_examples):
        test_x[i] = get_image(test_data[i])

    return np.array(train_x / 255.), np.array(dict_train['coarse_labels']), np.array(test_x / 255.), np.array(
        dict_test['coarse_labels'])


train_x, train_y, test_x, test_y = get_data()


class VGG(object):

    # def __init__(self,config):
    #     self.input = tf.placeholder(tf.float32, [None, D, D, 3],name='input')
    #     self.w = {}
    #     self.b = {}
    #     self.layers = {}
    #     pre_layer = self.input
    #     initializer = tf.contrib.layers.xavier_initializer()
    #     for i in range(len(config)):
    #         type,kernel,channels = config[i]
    #         if type == types['conv']:
    #             with tf.name_scope('conv%d'%(i+1)):
    #                 w = tf.Variable(initializer([kernel,kernel,pre_layer.shape[-1].value,channels]),name='w%d'%(i+1))
    #                 b = tf.Variable(tf.constant(0.1, tf.float32, [channels]),name='b%d'%(i+1))
    #                 layer = tf.nn.relu(tf.nn.conv2d(pre_layer, w, strides=[1, 1, 1, 1], padding='SAME') + b,name='conv%d'%(i+1))
    #             self.w['w%d'%(i+1)] = w
    #             self.b['b%d'%(i+1)] = b
    #             self.layers['l%d'%(i+1)] = layer
    #             pre_layer = layer
    #         elif type == types['pool']:
    #             layer = tf.nn.max_pool(pre_layer,ksize=[1,kernel,kernel,1],strides=[1,2,2,1],padding='SAME',name='pool%d'%(i+1))
    #             self.layers['l%d'%(i+1)] = layer
    #             pre_layer = layer
    #         else:
    #             #fc
    #             with tf.name_scope('fc%d'%(i+1)):
    #                 pre_layer_size = np.prod((pre_layer.shape[1:])).value
    #                 pre_reshape = tf.reshape(pre_layer, shape=[-1, pre_layer_size])
    #                 w = tf.Variable(initializer([pre_layer_size,channels]),name='w%d'%(i+1))
    #                 b = tf.Variable(tf.constant(0.1, tf.float32, [channels]),name='b%d'%(i+1))
    #                 #need to add relu activation except the last layer
    #                 layer = tf.add(tf.matmul(pre_reshape, w),b,name='fc%d'%(i+1))
    #             self.w['w%d'%(i+1)] = w
    #             self.b['b%d'%(i+1)] = b
    #             self.layers['l%d'%(i+1)] = layer
    #             pre_layer = layer
    #
    #     # add softmax layer
    #     self.out = tf.nn.softmax(pre_layer)

    def __init__(self,pre_clean=True):


        self.num_classes = 100
        self.input = tf.placeholder(tf.float32, [None, D, D, 3], name='input')
        self.layers = []
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
        self.test_hist=[]

        self.labels = tf.placeholder(tf.int32, [None, self.num_classes], name='labels')

        # aug_in = tf.image.flip_left_right(self.input)
        # total_in = tf.concat([self.input,aug_in],axis=0,name='total_input')

        self.layers.append(tf.layers.conv2d(self.input, 64, 3, padding='same', activation=tf.nn.relu,
                                            kernel_regularizer=self.regularizer, name='conv1'))
        self.layers.append(
            tf.layers.conv2d(self.layers[-1], 64, 3, padding='same', activation=tf.nn.relu,
                             kernel_regularizer=self.regularizer, name='conv2'))

        self.layers.append(tf.layers.max_pooling2d(self.layers[-1], 2, 2, padding='same', name='pool1'))

        self.layers.append(
            tf.layers.conv2d(self.layers[-1], 128, 3, padding='same', activation=tf.nn.relu,
                             kernel_regularizer=self.regularizer, name='conv3'))
        self.layers.append(
            tf.layers.conv2d(self.layers[-1], 128, 3, padding='same', activation=tf.nn.relu,
                             kernel_regularizer=self.regularizer, name='conv4'))

        self.layers.append(tf.layers.max_pooling2d(self.layers[-1], 2, 2, padding='same', name='pool2'))

        self.layers.append(
            tf.layers.conv2d(self.layers[-1], 256, 3, padding='same', activation=tf.nn.relu,
                             kernel_regularizer=self.regularizer, name='conv5'))
        self.layers.append(
            tf.layers.conv2d(self.layers[-1], 256, 3, padding='same', activation=tf.nn.relu,
                             kernel_regularizer=self.regularizer, name='conv6'))
        self.layers.append(
            tf.layers.conv2d(self.layers[-1], 256, 1, padding='same', activation=tf.nn.relu,
                             kernel_regularizer=self.regularizer, name='conv7'))

        self.layers.append(tf.layers.max_pooling2d(self.layers[-1], 2, 2, padding='same', name='pool3'))

        self.layers.append(
            tf.layers.conv2d(self.layers[-1], 512, 3, padding='same', activation=tf.nn.relu,
                             kernel_regularizer=self.regularizer, name='conv8'))
        self.layers.append(
            tf.layers.conv2d(self.layers[-1], 512, 3, padding='same', activation=tf.nn.relu,
                             kernel_regularizer=self.regularizer, name='conv9'))
        self.layers.append(
            tf.layers.conv2d(self.layers[-1], 512, 1, padding='same', activation=tf.nn.relu,
                             kernel_regularizer=self.regularizer, name='conv10'))

        self.layers.append(tf.layers.max_pooling2d(self.layers[-1], 2, 2, padding='same', name='pool4'))

        self.layers.append(
            tf.layers.conv2d(self.layers[-1], 512, 3, padding='same', activation=tf.nn.relu,
                             kernel_regularizer=self.regularizer, name='conv11'))
        self.layers.append(
            tf.layers.conv2d(self.layers[-1], 512, 3, padding='same', activation=tf.nn.relu,
                             kernel_regularizer=self.regularizer, name='conv12'))
        self.layers.append(
            tf.layers.conv2d(self.layers[-1], 512, 1, padding='same', activation=tf.nn.relu,
                             kernel_regularizer=self.regularizer, name='conv13'))

        self.layers.append(tf.layers.max_pooling2d(self.layers[-1], 2, 2, padding='same', name='pool5'))

        pre_layer_reshaped = tf.layers.flatten(self.layers[-1])
        self.layers.append(tf.layers.dropout(pre_layer_reshaped, rate=0.5, name='drop1'))
        self.layers.append(
            tf.layers.dense(self.layers[-1], units=4096, activation=tf.nn.relu, kernel_regularizer=self.regularizer,
                            name='fc1'))
        self.layers.append(tf.layers.dropout(self.layers[-1], rate=0.5, name='drop2'))
        self.layers.append(
            tf.layers.dense(self.layers[-1], units=4096, activation=tf.nn.relu, kernel_regularizer=self.regularizer,
                            name='fc2'))
        self.layers.append(
            tf.layers.dense(self.layers[-1], units=self.num_classes, kernel_regularizer=self.regularizer, name='fc3'))

        self.logits = self.layers[-1]
        self.predictions = {
            'classes': tf.argmax(input=self.logits, axis=1),
            'probs': tf.nn.softmax(self.logits, name='softmax_tensor')
        }

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
        l2_loss = tf.losses.get_regularization_loss()
        self.total_loss = tf.reduce_mean(cross_entropy) * 100 + l2_loss

        # accuracy of the trained model, between 0 (worst) and 1 (best)
        self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        # training step, the learning rate is a placeholder
        # the learning rate is: # 0.0001 + 0.003 * (1/e)^(step/2000)), i.e. exponential decay from 0.003->0.0001
        self.initial_lr = 1e-4 #+ tf.train.exponential_decay(1e-2, tf.train.get_global_step(), 4000, 1 / math.e)
        self.lr_tensor = tf.Variable(self.initial_lr, trainable=False)
        # self.train_step = tf.train.AdamOptimizer(self.lr).minimize(cross_entropy)
        self.train_step = tf.train.MomentumOptimizer(learning_rate=self.lr_tensor, momentum=0.9).minimize(cross_entropy)

        if pre_clean:
            dirs = ['./graphs/train','./graphs/test','./my_modules']
            for dir in dirs:
                for file in os.listdir(dir):
                    os.remove(os.path.join(dir, file))

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.writer_train = tf.summary.FileWriter('./graphs/train', tf.get_default_graph())
        self.writer_test = tf.summary.FileWriter('./graphs/test', tf.get_default_graph())

        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('loss', self.total_loss)
        self.merged = tf.summary.merge_all()
        self.saver = tf.train.Saver()
        # self.saver.restore(self.sess,tf.train.latest_checkpoint('./my_modules/'))

        # self.loss_tensor = tf.losses.sparse_softmax_cross_entropy(labels=self.labels, logits=self.logits)

        # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        # step = tf.train.get_global_step()
        # train_op = self.optimizer.minimize(loss=self.loss_tensor, global_step=step)

    def shuffle_data(self, x, y):
        # shuffle the train data
        train = zip(x, y)
        np.random.shuffle(train)
        x, y = zip(*train)
        x = np.array(x)
        y = np.array(y)
        return x, y

    def convert_to_1_hot_vec(self, y):
        size = y.size
        batch_y = np.zeros((size, self.num_classes))
        batch_y[np.arange(size), y] = 1
        return batch_y

    def get_batch_with_augmentation(self,batch_x,batch_y):
        batch_x = batch_x.astype(np.float32)
        batch_x_flipped = batch_x[:, :, ::-1, :]
        total_x = np.concatenate([batch_x, batch_x_flipped])
        batch_y = self.convert_to_1_hot_vec(batch_y)
        total_y = np.concatenate([batch_y, batch_y])
        return total_x,total_y

    def train(self, train_x, train_y, test_x, test_y, batch_size, epochs=1):
        N = train_x.shape[0]
        num_steps = N / batch_size + 1
        for epoch in range(epochs):
            train_x, train_y = self.shuffle_data(train_x, train_y)
            for i in range(num_steps):
                # get batch
                # ind = np.random.choice(range(train_y.size),batch_size)
                size = batch_size
                ind = np.arange(size)
                if (N - i * batch_size) < batch_size:
                    size = N - i * batch_size
                    ind = np.arange(size) + N - size
                else:
                    ind += i * batch_size

                # batch_x,batch_y = self.get_batch_with_augmentation(train_x[ind],train_y[ind])
                batch_x = train_x[ind].astype(np.float32)
                batch_y = self.convert_to_1_hot_vec(train_y[ind])

                # train the step with backprop
                self.sess.run(self.train_step, feed_dict={self.input: batch_x, self.labels: batch_y})


            ind_train = np.random.choice(range(train_y.size), batch_size)
            # batch_x,batch_y = self.get_batch_with_augmentation(train_x[ind_train],train_y[ind_train])
            batch_x = train_x[ind_train].astype(np.float32)
            batch_y = self.convert_to_1_hot_vec(train_y[ind_train])

            train_summary = self.sess.run(self.merged,
                                          feed_dict={self.input: batch_x,
                                                     self.labels: batch_y})
            self.writer_train.add_summary(train_summary, epoch)

            ind_test = np.random.choice(range(test_y.size), batch_size)
            # batch_x,batch_y = self.get_batch_with_augmentation(test_x[ind_test],test_y[ind_test])
            batch_x = test_x[ind_test].astype(np.float32)
            batch_y = self.convert_to_1_hot_vec(test_y[ind_test])

            test_acc = self.sess.run(self.accuracy, feed_dict={self.input: batch_x, self.labels: batch_y})
            #decide if we want to change the lr
            # avg = np.mean(self.test_hist)
            # if test_acc < avg:
            #     self.lr_tensor.assign()
            # self.test_hist.append(test_acc)
            # if len(self.test_hist>3):
            #     self.test_hist.pop(0)

            test_summary = self.sess.run(self.merged,
                                         feed_dict={self.input: batch_x,
                                                    self.labels: batch_y})
            self.writer_test.add_summary(test_summary, epoch)
            print('epoch ' + str(epoch) + ' finished')
            if epoch % 10 == 0:
                # save checkpoint
                self.saver.save(self.sess, './my_modules/a', epoch)

            # print("epoch " + str(epoch) + ": accuracy:" + str(acc) + " loss: " + str(cost))

            # acc, cost = self.sess.run([self.accuracy, self.cross_entropy],
            #                           feed_dict={self.input: test_x, self.labels: self.convert_to_1_hot_vec(test_y)})
            # print("test accuracy: " + str(acc) + " test loss: " + str(cost))

    def close(self):
        self.writer_train.close()
        self.writer_test.close()
        self.sess.close()

def copyanything(src, dst):
    try:
        shutil.copytree(src, dst)
    except:
       print('failed to copy')

def save_exper(exp_name):
    dir_path = './experiments'
    dir_path = os.path.join(dir_path, exp_name)
    # os.mkdir(dir_path)
    copyanything('./graphs', dir_path)
    copyanything('./my_modules',os.path.join(dir_path,'module'))



# init2 = tf.local_variables_initializer()

# sess = tf.Session()
# config = [['conv',3,64],['conv',3,64],['pool',2,2],
#             ['conv',3,128],['conv',3,128],['pool',2,2],
#             ['conv',3,256],['conv',3,256],['conv',1,256],['pool',2,2],
#           ['conv', 3, 512], ['conv', 3, 512], ['conv', 1, 512], ['pool', 2, 2],
#           ['conv', 3, 512], ['conv', 3, 512], ['conv', 1, 512], ['pool', 2, 2],
#           ['fc',0,4096],['fc',0,4096],['fc',0,100]]
#
# net = VGG(config)

net = VGG()

net.train(train_x, train_y, test_x, test_y, 256, 5)

# inp = train_x[:10]
# labels = train_y[:10]
# probs = sess.run(net.predictions,feed_dict={net.input:inp,net.labels:labels})
# print(probs)

net.close()

save_exper('temp')

# ex_im = train_x[25]
# plt.imshow(ex_im)
# plt.show()
