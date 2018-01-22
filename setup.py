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
max_vocab = 200000

assert os.path.isfile(src_add)
assert os.path.isfile(tgt_add)

dump = 1
if dump == 0:
    src_word2id, src_id2word, src_emb = load_emb(src_add, embed_dim, max_vocab, "src") #243003
    save_dump(src_word2id, src_id2word, src_emb, 'src')
    tgt_word2id, tgt_id2word, tgt_emb = load_emb(tgt_add, embed_dim,max_vocab, "tgt") #872827
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

generator = get_minibatch(batch_size, 32)


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

                for j in range(5):
                    src_x, src_y, tgt_x, tgt_y = generator.__next__()
                    y = np.concatenate((src_y, tgt_y), axis=0)
                    feed_dict = {
                        emb_model.src_emb_ids: src_x,
                        emb_model.tgt_emb_ids: tgt_x,
                        emb_model.y: y,
                        emb_model.lr_rate: lr_rate,
                        emb_model.isOrthoUpdate: np.array(0)
                    }

                    # emb_model_wx, emb_model_x, emb_model_hh1, emb_model_hh1_b1, emb_model_hh1_ac, emb_model_hh1_lec, emb_model_hh1_dp, emb_model_disc_logits = sess.run([emb_model.wx,emb_model.x, emb_model.hh1,emb_model.hh1_b1,emb_model.hh1_ac, emb_model.hh1_lec, emb_model.hh1_dp,emb_model.disc_logits], feed_dict=feed_dict)
                    # print("emb_model_wx",emb_model_wx,"\nemb_model_x", emb_model_x, "\nemb_model_hh1", emb_model_hh1, "\nemb_model_hh1_b1", emb_model_hh1_b1,"\nemb_model_hh1_ac",emb_model_hh1_ac,"\nemb_model_hh1_lec",emb_model_hh1_lec, "\nemb_model_hh1_dp" , emb_model_hh1_dp, "\ndisc_logits",emb_model_disc_logits)
                    disc_loss, emb_model_sigm, emb_model_ce, emb_model_y = sess.run([emb_model.disc_loss, emb_model.sigm, emb_model.ce, emb_model.y], feed_dict=feed_dict)
                    print("disc_loss", disc_loss, "emb_model_sigm" , emb_model_sigm , "emb_model_ce", emb_model_ce)
                    input(":")

                src_x, src_y, tgt_x, tgt_y = generator.__next__()
                y = np.concatenate((src_y, tgt_y), axis=0)
                feed_dict = {
                    emb_model.src_emb_ids: src_x,
                    emb_model.tgt_emb_ids: tgt_x,
                    emb_model.y: y,
                    emb_model.lr_rate: lr_rate,
                    emb_model.isOrthoUpdate: np.array(0)
                }
                _, map_loss = sess.run([map_train_step, emb_model.map_loss], feed_dict=feed_dict)
                _ = sess.run([emb_model.orgonal_update_step])
                # print("\t", "#map_loss: ", map_loss)
                if( i % 1 == 0 ):
                    lr_rate = lr_rate*lr_decay
                    print("epoch:", i_epoch, " batch:", i, "  disc_loss:", disc_loss, "  map_loss:", map_loss)
                lr_rate = max(lr_rate, min_lr)

            save_model(sess, emb_model, src_emb, tgt_emb, src_id2word, tgt_id2word, 'en-es'+str(i_epoch))