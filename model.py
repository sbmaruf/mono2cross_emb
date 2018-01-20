import tensorflow as tf
tf.set_random_seed(100)

class model(object):

    def __init__(self, src, tgt, identity, beta):

        self.src_emb_ids = tf.placeholder(tf.int32, shape=[None], name="src_emb_ids")
        self.tgt_emb_ids = tf.placeholder(tf.int32, shape=[None], name="tgt_emb_ids")

        self.y = tf.placeholder(tf.float32, shape=[None, None], name="domain_id")
        self.y_invert = 1 - self.y

        self.isOrthoUpdate = tf.placeholder(tf.bool, [], name="isOrthoUpdate")
        self.lr_rate = tf.placeholder(dtype=tf.float32, shape=[], name="lr_rate")

        src_emb = tf.Variable(
            src,
            name="src_emb",
            dtype=tf.float32,
            trainable=False)
        src_lookup = tf.nn.embedding_lookup(
            src_emb, self.src_emb_ids, name="src_lookup")

        tgt_emb = tf.Variable(
            tgt,
            name="src_emb",
            dtype=tf.float32,
            trainable=False)
        tgt_lookup = tf.nn.embedding_lookup(
            tgt_emb, self.tgt_emb_ids, name="tgt_lookup")

        self.theta_M = []
        self.W_trans = tf.Variable(
            identity,
            name="trans",
            dtype=tf.float32)
        self.theta_M.append(self.W_trans)
        temp = tf.transpose(self.W_trans)
        temp1 = lambda: self.W_trans
        temp2 = lambda: tf.scalar_mul(1 + beta, self.W_trans) - tf.scalar_mul(beta, tf.matmul(self.W_trans,
                                        tf.matmul(temp, self.W_trans)))
        self.W_trans = tf.cond(self.isOrthoUpdate, temp2, temp1)



        trans_emb = tf.matmul(src_lookup, self.W_trans)
        self.x = tf.concat([trans_emb, tgt_lookup], axis=0)


        self.theta_D = []

        hid1 = tf.get_variable(name="hid1",
                                dtype=tf.float32,
                                shape=[src.shape[1], 2048])
        b_hid1 = tf.get_variable(name="b_hid1",
                                dtype=tf.float32,
                                shape=[2048],
                                initializer=tf.zeros_initializer())
        self.theta_D.append(hid1)
        self.theta_D.append(b_hid1)

        trans_emb_hid1 = tf.matmul(self.x, hid1) + b_hid1
        trans_emb_hid1 = tf.nn.leaky_relu(trans_emb_hid1)
        trans_emb_hid1 = tf.nn.dropout(trans_emb_hid1, .9)

        hid2 = tf.get_variable(name="hid2",
                                dtype=tf.float32,
                                shape=[2048, 2048])
        b_hid2 = tf.get_variable(name="b_hid2",
                                dtype=tf.float32,
                                shape=[2048],
                                initializer=tf.zeros_initializer())
        hid1_hid2 = tf.matmul(trans_emb_hid1, hid2) + b_hid2
        hid1_hid2 = tf.nn.leaky_relu(hid1_hid2)
        hid1_hid2 = tf.nn.dropout(hid1_hid2, .9)

        self.theta_D.append(hid2)
        self.theta_D.append(b_hid2)


        out = tf.get_variable(name="out",
                               dtype=tf.float32,
                               shape=[2048, 1])
        b_out = tf.get_variable(name="b_out",
                                 dtype=tf.float32,
                                 shape=[1],
                                 initializer=tf.zeros_initializer())
        self.disc_logits = tf.matmul(hid1_hid2, out) + b_out

        self.theta_D.append(out)
        self.theta_D.append(b_out)

        self.disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                            labels=self.y, logits=self.disc_logits))
        self.map_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                            labels=self.y_invert, logits=self.disc_logits))


