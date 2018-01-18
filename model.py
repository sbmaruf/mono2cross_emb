import tensorflow as tf
tf.set_random_seed(100)

class model(object):

    def __init__(self, src, tgt):
        self.src_emb_ids = tf.placeholder(tf.int32, shape=[None], name="src_emb_ids")
        self.tgt_emb_ids = tf.placeholder(tf.int32, shape=[None], name="tgt_emb_ids")
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

        W_trans = tf.eye(num_rows=src.shape[1],
                         num_columns=src.shape[1],
                         dtype=tf.float32,
                         name="trans")
        trans_emb = tf.multiply(src_lookup, W_trans)


        hid1 = tf.get_variable(name="hid1",
                                dtype=tf.float32,
                                shape=[src.shape[1], 2048])
        b_hid1 = tf.get_variable(name="b_hid1",
                                dtype=tf.float32,
                                shape=[2048],
                                initializer=tf.zeros_initializer())
        trans_emb_hid1 = tf.multiply(trans_emb, hid1) + b_hid1
        trans_emb_hid1 = tf.nn.leaky_relu(trans_emb_hid1)
        trans_emb_hid1 = tf.nn.dropout(trans_emb_hid1, .9)

        hid2 = tf.get_variable(name="hid2",
                                dtype=tf.float32,
                                shape=[2048, 2048])
        b_hid2 = tf.get_variable(name="b_hid2",
                                dtype=tf.float32,
                                shape=[2048],
                                initializer=tf.zeros_initializer())
        hid1_hid2 = tf.multiply(trans_emb_hid1, hid2) + b_hid2
        hid1_hid2 = tf.nn.leaky_relu(hid1_hid2)
        hid1_hid2 = tf.nn.dropout(hid1_hid2, .9)

        out = tf.get_variable(name="out",
                               dtype=tf.float32,
                               shape=[2048, 1])
        b_out = tf.get_variable(name="b_out",
                                 dtype=tf.float32,
                                 shape=[1],
                                 initializer=tf.zeros_initializer())
        hid2_out = tf.multiply(hid1_hid2, out) + b_out
        hid2_out = tf.nn.leaky_relu(hid2_out)
        hid2_out = tf.nn.dropout(hid2_out)

        logit = tf.nn.sigmoid(hid2_out, name="logit")
