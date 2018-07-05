import numpy as np
import tensorflow as tf
from market1501 import Market1501
import matplotlib.pyplot as plt
import losses
import os,shutil
import metrics
from tensorflow.contrib.tensorboard.plugins import projector


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# from tensorflow.contrib.framework.python.ops import add_arg_scope

tf.app.flags.DEFINE_string('inputs_path', './Market-1501-v15.09.15', 'path to input')
tf.app.flags.DEFINE_string('outputs_path', './graphs/vgg/DML', 'path to output')
tf.app.flags.DEFINE_string('dataset', 'Market1501', 'name of dataset')
tf.app.flags.DEFINE_integer('batch_size', 6, 'batch size')
tf.app.flags.DEFINE_integer('max_epochs', 1000, 'num of epochs')
tf.app.flags.DEFINE_integer('embedding_length', 5, 'embedding length')

FLAGS = tf.app.flags.FLAGS


# @add_arg_scope
def vgg_16(inputs, embedding_length):
    # TODO: implement VGG_16 - done but need to check
    with tf.name_scope('vgg_16'):
        layers = []
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
        layers.append(tf.layers.conv2d(inputs, 64, 3, padding='same', activation=tf.nn.relu,
                                       kernel_regularizer=regularizer, name='conv1'))
        layers.append(
            tf.layers.conv2d(layers[-1], 64, 3, padding='same', activation=tf.nn.relu,
                             kernel_regularizer=regularizer, name='conv2'))

        layers.append(tf.layers.max_pooling2d(layers[-1], 2, 2, padding='same', name='pool1'))

        layers.append(
            tf.layers.conv2d(layers[-1], 128, 3, padding='same', activation=tf.nn.relu,
                             kernel_regularizer=regularizer, name='conv3'))
        layers.append(
            tf.layers.conv2d(layers[-1], 128, 3, padding='same', activation=tf.nn.relu,
                             kernel_regularizer=regularizer, name='conv4'))

        layers.append(tf.layers.max_pooling2d(layers[-1], 2, 2, padding='same', name='pool2'))

        layers.append(
            tf.layers.conv2d(layers[-1], 256, 3, padding='same', activation=tf.nn.relu,
                             kernel_regularizer=regularizer, name='conv5'))
        layers.append(
            tf.layers.conv2d(layers[-1], 256, 3, padding='same', activation=tf.nn.relu,
                             kernel_regularizer=regularizer, name='conv6'))
        layers.append(
            tf.layers.conv2d(layers[-1], 256, 1, padding='same', activation=tf.nn.relu,
                             kernel_regularizer=regularizer, name='conv7'))

        layers.append(tf.layers.max_pooling2d(layers[-1], 2, 2, padding='same', name='pool3'))

        layers.append(
            tf.layers.conv2d(layers[-1], 512, 3, padding='same', activation=tf.nn.relu,
                             kernel_regularizer=regularizer, name='conv8'))
        layers.append(
            tf.layers.conv2d(layers[-1], 512, 3, padding='same', activation=tf.nn.relu,
                             kernel_regularizer=regularizer, name='conv9'))
        layers.append(
            tf.layers.conv2d(layers[-1], 512, 1, padding='same', activation=tf.nn.relu,
                             kernel_regularizer=regularizer, name='conv10'))

        layers.append(tf.layers.max_pooling2d(layers[-1], 2, 2, padding='same', name='pool4'))

        layers.append(
            tf.layers.conv2d(layers[-1], 512, 3, padding='same', activation=tf.nn.relu,
                             kernel_regularizer=regularizer, name='conv11'))
        layers.append(
            tf.layers.conv2d(layers[-1], 512, 3, padding='same', activation=tf.nn.relu,
                             kernel_regularizer=regularizer, name='conv12'))
        layers.append(
            tf.layers.conv2d(layers[-1], 512, 1, padding='same', activation=tf.nn.relu,
                             kernel_regularizer=regularizer, name='conv13'))

        pool5 = tf.layers.max_pooling2d(layers[-1], 2, 2, padding='same', name='pool5')
        layers.append(pool5)

        fc1 = tf.layers.conv2d(
            pool5, 512, 7, name='fc1', activation=tf.nn.relu, padding='same', kernel_regularizer=regularizer
        )
        dp1 = tf.layers.dropout(fc1, rate=0.5, name='drop1')
        fc2 = tf.layers.conv2d(
            dp1, 512, 1, name='fc2', activation=tf.nn.relu, padding='same', kernel_regularizer=regularizer
        )
        g_pool = tf.reduce_mean(fc2, [1, 2], keep_dims=True, name='global_pool')
        dp2 = tf.layers.dropout(g_pool, rate=0.5, name='drop2')
        output = tf.layers.conv2d(dp2, embedding_length, [1, 1], activation=None)

    return output


def read_and_prepare_data(dataset):
    X_train, Y_train, _ = dataset.read_train()
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_val, Y_val, _ = dataset.read_validation()
    X_val = np.array(X_val)
    Y_val = np.array(Y_val)
    return X_train, Y_train, X_val, Y_val


def create_opt(net, labels, global_step, decay_steps=30000, learning_rate_decay_factor=0.5, learning_rate=0.01):
    lr = tf.train.exponential_decay(learning_rate,
                                    global_step,
                                    decay_steps,
                                    learning_rate_decay_factor,
                                    staircase=True,
                                    name='exponential_decay_learning_rate')

    features = tf.squeeze(net, axis=[1, 2])
    # dml_loss,classes_mean,var = losses.magnet_loss(features, labels)
    dml_loss = losses.triplet_loss(features, labels,False)
    l2_loss = tf.losses.get_regularization_loss()
    loss = dml_loss + l2_loss
    opt = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss, global_step=global_step)
    return opt, loss, lr


def predict(net, y):
    with tf.name_scope('predict'):
        tau = 1e-6
        squeezed = tf.squeeze(net, axis=[1, 2])
        distances = metrics.cosine_distance(squeezed)
        # distances = tf.Print(distances, [distances], "distances: ",first_n=1,summarize=50)
        # y == y.T
        y_identities = tf.cast(tf.equal(tf.expand_dims(y, 0), tf.expand_dims(y, 1),name='y_ident'), tf.float32)
        # y_identities = tf.Print(y_identities,[y_identities],"y_ident: ",first_n=1,summarize=100)
        dist_identites = tf.cast(tf.less(distances, tau,name='dist_ident'), tf.float32)
        # dist_identites = tf.Print(dist_identites,[dist_identites],"dist_identites: ",first_n=1,summarize=100)

        res_matrix = tf.cast(tf.equal(y_identities, dist_identites,name='res_matrix'), tf.float32)
        # res_matrix = tf.Print(res_matrix,[res_matrix],"res_matrix: ",first_n=1,summarize=100)
        #calc accuracy - atm it is wrong calculation! TODO:fix this
        num_of_elements = (np.square(FLAGS.batch_size)+FLAGS.batch_size)*0.5
        # including the classification of example i is equal to itself
        accuracy = tf.divide(tf.reduce_sum(tf.matrix_band_part(res_matrix, 0, -1)),num_of_elements)
        #accuracy = tf.Print(accuracy,[accuracy,tf.matrix_band_part(res_matrix, 0, -1)],"accuracy: ",summarize=100)
    # TODO: implement predict
    pred = None
    return pred, accuracy

def calc_tau(net,y):
    squeezed = tf.squeeze(net, axis=[1, 2])
    distances = metrics.cosine_distance(squeezed)
    y_identities = tf.cast(tf.equal(tf.expand_dims(y, 0), tf.expand_dims(y, 1), name='y_ident'), tf.float32)


def main():
    #delete files
    shutil.rmtree(FLAGS.outputs_path)

    # placeholders and variables
    inputs = tf.placeholder(tf.float32, shape=[None, None, None, 3])
    y = tf.placeholder(tf.int32, shape=[None])
    is_training = tf.placeholder(tf.bool)
    gstep = tf.Variable(0, dtype=tf.int32,
                        trainable=False, name='global_step')

    # read dataset
    dataset = Market1501(FLAGS.inputs_path, num_validation_y=0.1, seed=1234)
    X_train, Y_train, X_val, Y_val = read_and_prepare_data(dataset)


    # create network and opt
    net = vgg_16(inputs, FLAGS.embedding_length)
    labels = y
    opt, loss, lr = create_opt(net, y, gstep)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('learning_rate', lr)
    pred, acc = predict(net, labels)
    tf.summary.scalar('accuracy',acc)

    EMB_DIR = FLAGS.outputs_path + '/emb'

    # summary
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.outputs_path + '/train', tf.get_default_graph())
    test_writer = tf.summary.FileWriter(FLAGS.outputs_path + '/test')
    emb_writer = tf.summary.FileWriter(EMB_DIR)
    saver = tf.train.Saver()

    # embedding
    embedding_batch = 100
    features = tf.squeeze(net, axis=[1, 2])
    embedding_var = tf.Variable(np.zeros((embedding_batch,FLAGS.embedding_length)), name='vis_var')
    config = projector.ProjectorConfig()
    #### embedding = config.embeddings.add()
    #### embedding.tensor_name = embedding_var.name
    # Specify where you find the metadata
    #### embedding.metadata_path = os.path.join(EMB_DIR, 'metadata.tsv')
    # Say that you want to visualise the embeddings
    projector.visualize_embeddings(emb_writer, config)

    # running network
    sess = tf.InteractiveSession()
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    step = gstep.eval()

    # FF
    net_features = sess.run(features, feed_dict={inputs: b, y: l_b, is_training: True})

    # create the data for visualization

    images = tf.Variable(features, name='images')
    x_tsv, y_tsv = X_train[:embedding_batch], Y_train[:embedding_batch]
    metadata_path = os.path.join(EMB_DIR, 'metadata.tsv')
    with open(metadata_path, 'w') as f:
        #f.write("Index\tLabel\n")
        for index, label in enumerate(y_tsv):
            f.write("%d\n" % label)

    with tf.Session() as sess:
        saver = tf.train.Saver([images])

        sess.run(images.initializer)
        saver.save(sess, os.path.join(EMB_DIR, 'images.ckpt'))

        config = projector.ProjectorConfig()
        # One can add multiple embeddings.
        embedding = config.embeddings.add()
        embedding.tensor_name = images.name
        embedding.metadata_path = metadata_path
        # Link this tensor to its metadata file (e.g. labels).
        # Saves a config file that TensorBoard will read during startup.
        projector.visualize_embeddings(tf.summary.FileWriter(EMB_DIR), config)

    #### net_features = sess.run(features, feed_dict={inputs: x_tsv, y: y_tsv})
    #### sess.run(embedding_var.assign(net_features))
    #### saver.save(sess, os.path.join(EMB_DIR, "model.ckpt"), 1)


    # train network
    for epoch in range(FLAGS.max_epochs):
        print('Epoch: {}'.format(epoch))
        permutation = np.random.permutation(Y_train.shape[0])
        x_train = X_train[permutation]
        y_train = Y_train[permutation]
        for batch in range(X_train.shape[0] / FLAGS.batch_size):
            b = x_train[batch * FLAGS.batch_size: (batch + 1) * FLAGS.batch_size]

            l_b = y_train[batch * FLAGS.batch_size: (batch + 1) * FLAGS.batch_size]
            _, loss_value, _merged = sess.run([opt, loss, merged], feed_dict={inputs: b, y: l_b, is_training: True})
            if batch % 100 == 0:
                print("step {} - loss: {}".format(step, loss_value))
            step += 1
            train_writer.add_summary(_merged, step)

        if (epoch % 20) == 0:
            saver.save(sess, FLAGS.outputs_path + "/weights", step)
            break


if __name__ == "__main__":
    main()
