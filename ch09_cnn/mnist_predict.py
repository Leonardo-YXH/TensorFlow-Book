import tensorflow as tf
import numpy as np

def deep_nn(x, W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2, keep_prob):
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv

def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
class mnist_data_predict(object):
    def __init__(self,meta_graph='./log/my_model.ckpt.meta',checkpoint_path='./log'):
        self.sess=tf.Session()
        tf.train.import_meta_graph(meta_graph).restore(self.sess, tf.train.latest_checkpoint(checkpoint_path))
        self.graph=tf.get_default_graph()
        self.W_conv1 = get_w_b(self.sess, self.graph, 'conv1/W_conv1:0')
        self.b_conv1 = get_w_b(self.sess, self.graph, 'conv1/b_conv1:0')
        self.W_conv2 = get_w_b(self.sess, self.graph, 'conv2/W_conv2:0')
        self.b_conv2 = get_w_b(self.sess, self.graph, 'conv2/b_conv2:0')

        self.W_fc1 = get_w_b(self.sess, self.graph, 'fc1/W_fc1:0')
        self.b_fc1 = get_w_b(self.sess, self.graph, 'fc1/b_fc1:0')
        self.W_fc2 = get_w_b(self.sess, self.graph, 'fc2/W_fc2:0')
        self.b_fc2 = get_w_b(self.sess, self.graph, 'fc2/b_fc2:0')

    def predict(self,image):
        y_conv = deep_nn(image, self.W_conv1, self.b_conv1, self.W_conv2, self.b_conv2, self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2, 1.0)
        y = tf.argmax(y_conv, 1)
        print(self.sess.run(y))
    def close(self):
        self.sess.close()


def get_w_b(sess, graph, name):
    return sess.run(graph.get_tensor_by_name(name))

if __name__=="__main__":
    pass