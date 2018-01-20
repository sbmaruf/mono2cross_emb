import os
from utils import load_emb, \
    getOptimizer, \
    grad_compute, \
    get_minibatch,\
    save_dump,\
    load_dump,\
    save_model

from model import *
import time
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

src_add = "./data/sskip.100.vectors"
tgt_add = "./data/ES64"
embed_dim = 100
beta = .001

assert os.path.isfile(src_add)
assert os.path.isfile(tgt_add)

dump = 1
if dump == 0:
    src_word2id, src_id2word, src_emb = load_emb(src_add, embed_dim) #243003
    save_dump(src_word2id, src_id2word, src_emb, 'src')
    tgt_word2id, tgt_id2word, tgt_emb = load_emb(tgt_add, embed_dim) #872827
    save_dump(tgt_word2id, tgt_id2word, tgt_emb, 'tgt')
else:
    src_word2id, src_id2word, src_emb = load_dump('src')
    tgt_word2id, tgt_id2word, tgt_emb = load_dump('tgt')

num_of_epoch = 5
num_of_batch = 1000000
batch_size = 32
lr_rate = .1
lr_decay = .98
min_lr = 0.0000001

generator = get_minibatch(len(src_emb), len(tgt_emb), batch_size)

tf.reset_default_graph()
my_graph = tf.Graph()
with my_graph.as_default():
    sess = tf.Session()
    with sess.as_default():

        emb_model = model(src_emb, tgt_emb, np.identity(src_emb.shape[1]), beta )

        disc_optimizer = getOptimizer(0, emb_model.lr_rate, 'disc: ')
        map_optimizer = getOptimizer(0, emb_model.lr_rate, 'map: ')

        map_train_step, \
        disc_train_step = grad_compute(map_optimizer,
                                       disc_optimizer,
                                       emb_model)
        init = tf.global_variables_initializer()
        sess.run(init)
        for i_epoch in range(num_of_epoch):
            print("Starting adversarial training epoch {} ... .. .".format(i_epoch))
            tic = time.time()
            for i in range(num_of_batch):
                epoch, src_x, src_y, tgt_x, tgt_y = generator.__next__()
                y = np.concatenate((src_y, tgt_y), axis=0)
                feed_dict = {
                    emb_model.src_emb_ids: src_x,
                    emb_model.tgt_emb_ids: tgt_x,
                    emb_model.y: y,
                    emb_model.isOrthoUpdate: np.array(0),
                    emb_model.lr_rate: lr_rate
                }
                _, _, disc_loss, map_loss = sess.run([disc_train_step, map_train_step,
                                            emb_model.disc_loss, emb_model.map_loss], feed_dict=feed_dict)
                feed_dict = {
                    emb_model.isOrthoUpdate: np.array(1)
                }
                _ = sess.run([emb_model.W_trans], feed_dict=feed_dict)
                if( i % 500 == 0 ):
                    lr_rate = lr_rate*lr_decay
                    print("epoch:", i_epoch, " batch:", i, " dosc_loss:", disc_loss, "  map_loss:", map_loss)
                lr_rate = max(lr_rate, min_lr)
                save_model(sess, emb_model, src_emb, tgt_emb, src_id2word, tgt_id2word, 'en-es'+str(i))