from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

def load_gray_scale_data(pickle_file = 'notMNIST.pickle'):
  with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)
  return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels

def reformat(dataset, labels, image_size, num_labels, num_channels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

def load_reformat_data(image_size, num_labels, num_channels, pickle_file = 'notMNIST.pickle'):
  print("Loading gray scale data from pickle file")
  train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = load_gray_scale_data(pickle_file)
  print("Reformat data into good shape")
  train_dataset, train_labels = reformat(train_dataset, train_labels, image_size, num_labels, num_channels)
  valid_dataset, valid_labels = reformat(valid_dataset, valid_labels, image_size, num_labels, num_channels)
  test_dataset, test_labels = reformat(test_dataset, test_labels, image_size, num_labels, num_channels)
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)
  return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

if __name__=='__main__':

  num_steps = 2001

  image_size = 28
  num_labels = 10
  num_channels = 1

  train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = load_reformat_data(image_size, num_labels, num_channels, 'notMNIST.pickle')
  
  batch_size = 16
  patch_size = 5
  depth = 16
  num_hidden = 64
  ksize = 2
  pool_stride = 2
  dropout = 0.9

  graph = tf.Graph()
  with graph.as_default():

    # Input data.
    tf_train_dataset = tf.placeholder(
      tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    # Variables.
    c1_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, num_channels, depth], stddev=0.1))
    c1_biases = tf.Variable(tf.zeros([depth]))
    c3_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, depth, depth], stddev=0.1))
    c3_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
    c5_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, depth, depth], stddev=0.1))
    c5_biases = tf.Variable(tf.constant(1.0, shape=[depth]))

    f6_weights = tf.Variable(tf.truncated_normal(
        [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
    f6_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
    out_weights = tf.Variable(tf.truncated_normal(
        [num_hidden, num_labels], stddev=0.1))
    out_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
    # dropout probability
    keep_prob = tf.placeholder(tf.float32) 
    # weight decay
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.08, global_step, 100, 0.9)

    # Model.
    def model(data, data_type='train'):
      # c1
      c1 = tf.nn.conv2d(data, c1_weights, [1, 1, 1, 1], padding='SAME')
      c1 = tf.nn.relu(c1 + c1_biases)
      # s2
      s1 = tf.nn.max_pool(c1, [1, ksize, ksize, 1], [1, pool_stride, pool_stride, 1], padding='SAME')
      # c3
      c3 = tf.nn.conv2d(s1, c3_weights, [1, 1, 1, 1], padding='SAME')
      c3 = tf.nn.relu(c3 + c3_biases)
      # s4
      s4 = tf.nn.max_pool(c3, [1, ksize, ksize, 1], [1, pool_stride, pool_stride,1], padding='SAME')
      # c5
      c5 = tf.nn.conv2d(s4, c5_weights, [1, 1, 1, 1], padding='SAME')
      c5 = tf.nn.relu(c5 + c5_biases)
      # reshape the output of c5
      shape = c5.get_shape().as_list()
      reshape = tf.reshape(c5, [shape[0], shape[1] * shape[2] * shape[3]])
      if data_type == 'train':
        reshape = tf.nn.dropout(reshape, keep_prob)
      # F6
      f6 = tf.nn.relu(tf.matmul(reshape, f6_weights) + f6_biases)
      if data_type == 'train':
        f6 = tf.nn.dropout(f6, keep_prob)
      # output
      return tf.matmul(f6, out_weights) + out_biases
  
    # Training computation.
    logits = model(tf_train_dataset, 'train')
    loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
    
    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
  
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset, 'valid'))
    test_prediction = tf.nn.softmax(model(tf_test_dataset, 'test'))


  # start training networks
  with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    for step in range(num_steps):
      offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
      batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
      batch_labels = train_labels[offset:(offset + batch_size), :]
      feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob : dropout}
      _, l, predictions = session.run(
        [optimizer, loss, train_prediction], feed_dict=feed_dict)
      if (step % 50 == 0):
        print('Minibatch loss at step %d: %f' % (step, l))
        print('Learning rate at step %d: %f' % (step, learning_rate.eval()))
        print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
        print('Validation accuracy: %.1f%%' % accuracy(
          valid_prediction.eval(), valid_labels))
    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

