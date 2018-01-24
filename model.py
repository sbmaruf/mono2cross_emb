import tensorflow as tf
tf.set_random_seed(100)
import numpy as np

class model(object):

    def __init__(self, src, tgt, identity, beta, dropout_map, dropout_hid):
        with tf.variable_scope("inputs-of-the-model"):
            self.src_emb_ids = tf.placeholder(tf.int32, shape=[None], name="src_emb_ids")
            self.tgt_emb_ids = tf.placeholder(tf.int32, shape=[None], name="tgt_emb_ids")

            self.y = tf.placeholder(tf.float32, shape=[None, None], name="domain_id")
            self.y_invert = 1 - self.y

            self.lr_rate = tf.placeholder(dtype=tf.float32, shape=[], name="lr_rate")


        with tf.variable_scope("source-embedding"):
            src_emb = tf.Variable(
                src,
                name="src_emb",
                dtype=tf.float32,
                trainable=False)
            self.src_lookup = tf.nn.embedding_lookup(
                src_emb, self.src_emb_ids, name="src_lookup")


        with tf.variable_scope("target-embedding"):
            tgt_emb = tf.Variable(
                tgt,
                name="src_emb",
                dtype=tf.float32,
                trainable=False)
            self.tgt_lookup = tf.nn.embedding_lookup(
                tgt_emb, self.tgt_emb_ids, name="tgt_lookup")



        self.theta_M = []
        with tf.variable_scope("mapping-matrix"):
            self.W_trans = tf.Variable(
                identity,
                name="trans",
                dtype=tf.float32)
            self.theta_M.append(self.W_trans)
            self.w_shape = tf.shape(self.W_trans)

            temp = tf.transpose(self.W_trans)
            self.orthogonal_update_step = tf.assign(self.W_trans, tf.scalar_mul(1 + beta, self.W_trans) -
                                tf.scalar_mul(beta, tf.matmul(self.W_trans, tf.matmul(temp, self.W_trans))))



        with tf.variable_scope("projection"):
            self.trans_emb = tf.matmul(self.src_lookup, self.W_trans)

            self.x = tf.concat([self.trans_emb, self.tgt_lookup], axis=0)
            self.x = tf.nn.dropout(self.x, dropout_map)



        self.theta_D = []
        with tf.variable_scope("hiddedn-layer-1"):

            self.hid1 = tf.get_variable(name="hid1",
                                    dtype=tf.float32,
                                    shape=[src.shape[1], 2048])
            # self.hid1 = tf.Variable(np.ones((src.shape[1], 2048)), dtype=tf.float32, name="hid1")
            self.b_hid1 = tf.get_variable(name="b_hid1",
                                    dtype=tf.float32,
                                    shape=[2048],
                                    initializer=tf.zeros_initializer())
            self.theta_D.append(self.hid1)
            self.theta_D.append(self.b_hid1)

            self.x_hid1 = tf.matmul(self.x, self.hid1) + self.b_hid1
            self.x_hid1_ac = tf.nn.leaky_relu(self.x_hid1)
            self.x_hid1_dp = tf.nn.dropout(self.x_hid1_ac, dropout_hid)


        with tf.variable_scope("hidden-layer-2"):
            self.hid2 = tf.get_variable(name="hid2",
                                    dtype=tf.float32,
                                    shape=[2048, 2048])
            # self.hid2 = tf.Variable(np.ones((2048, 2048)), dtype=tf.float32, name="hid2")
            self.b_hid2 = tf.get_variable(name="b_hid2",
                                    dtype=tf.float32,
                                    shape=[2048],
                                    initializer=tf.zeros_initializer())
            self.theta_D.append(self.hid2)
            self.theta_D.append(self.b_hid2)


            self.hid1_hid2 = tf.matmul(self.x_hid1_dp, self.hid2) + self.b_hid2
            self.hid1_hid2_ac = tf.nn.leaky_relu(self.hid1_hid2)
            self.hid1_hid2_dp = tf.nn.dropout(self.hid1_hid2_ac, dropout_hid)



        with tf.variable_scope("output-layer"):
            self.out = tf.get_variable(name="out",
                                   dtype=tf.float32,
                                   shape=[2048, 1])
            # self.out = tf.Variable(np.ones((2048, 1)), dtype=tf.float32, name="out")
            self.b_out = tf.get_variable(name="b_out",
                                     dtype=tf.float32,
                                     shape=[1],
                                     initializer=tf.zeros_initializer())
            self.theta_D.append(self.out)
            self.theta_D.append(self.b_out)
            self.disc_logits = tf.matmul(self.hid1_hid2_dp, self.out) + self.b_out
            self.sigm = tf.nn.sigmoid(self.disc_logits)



        with tf.variable_scope("loss-calculation"):
            self.disc_loss = -tf.reduce_mean(self.y * (tf.log(self.sigm+1e-12))+ (1-self.y)*(tf.log(1-self.sigm+1e-12)))
            self.map_loss = -tf.reduce_mean(self.y_invert * (tf.log(self.sigm + 1e-12)) + (1 - self.y_invert) * (tf.log(1 - self.sigm + 1e-12)))

