import tensorflow as tf
import numpy as np
import time

def load_emb(src_add, embed_dim):
    """
    Reload pretrained embeddings from a text file.
    """
    word2id = {}
    vectors = []
    print("loading pretrained-vector(s) ... .. .")

    # load pretrained embeddings
    with open(src_add) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            if(not (vect.shape[0] == 64 or vect.shape[0] == 100 ) ):
                print("error loading vector")
                input("here")
                continue
            if(vect.shape == (64,) and embed_dim == 100 ):
                vect = np.append(vect, np.zeros(36))
            if np.linalg.norm(vect) == 0:
                vect[0] = 0.01
            assert word not in word2id
            assert vect.shape[0] == 100
            if( i%1000 == 0 ):
                print(i)
            word2id[word] = len(word2id)
            vectors.append(vect[None])


    print("Loaded {0} pre-trained word embeddings".format(len(vectors)))

    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.concatenate(vectors, 0)

    return word2id, id2word, embeddings

# def loademb(emb_param, embed_dim):
#     """
#     Load the embedding vector from the dataset.
#     """
#
#     print("Loading pretrained vector(s) ... .. .")
#     tic = time.time()
#     Word2Vec={}
#     itr = 0
#     address = emb_param
#     for line in open(address, 'r', 'utf8'):
#         line = line.rstrip().split()
#         if not (len(line)==65 or len(line)==101):
#             continue
#         k = []
#         if len(line) == 65:
#             temp = 64
#         else:
#             temp = embed_dim
#         for i in range(temp):
#             k.append(float(line[i+1]))
#         while( len(k) < embed_dim ):
#             k.append(0.0)
#         Word2Vec[line[0]] = k
#         itr += 1
#
#     print("total vector in init_emb", itr)
#     toc = time.time()
#     print("skip-gram vector loading time ", toc-tic , " (s)")
#     return Word2Vec


def num2optimizer(i):
    if (i == 0):
        return "SGD"
    elif (i == 1):
        return "Adadelta"
    elif (i == 2):
        return "Adagrad"
    elif (i == 3):
        return "AdagradDA"
    elif (i == 4):
        return "Momentum"
    elif (i == 5):
        return "Adam"
    elif (i == 6):
        return "ftrl"
    elif (i == 7):
        return "ProximalSGD"
    elif (i == 8):
        return "ProximalAdagrad"
    elif (i == 9):
        return "RMSProp"
    else:
        return None


def getOptimizer(learning_method, learning_rate):
    assert (learning_method >= 0
            and learning_method <= 9
            and type(learning_method) == type(int(1)))
    print("Using optimizer ", num2optimizer(learning_method))
    if (learning_method == 0):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    elif (learning_method == 1):
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
    elif (learning_method == 2):
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    elif (learning_method == 3):
        optimizer = tf.train.AdagradDAOptimizer(learning_rate=learning_rate)
    elif (learning_method == 4):
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate)
    elif (learning_method == 5):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    elif (learning_method == 6):
        optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate)
    elif (learning_method == 7):
        optimizer = tf.train.ProximalGradientDescentOptimizer(learning_rate=learning_rate)
    elif (learning_method == 8):
        optimizer = tf.train.ProximalAdagradOptimizer(learning_rate=learning_rate)
    else:
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    return optimizer



def grad_compute(map_optimizer, disc_optimizer, model ):
    map_train_step = map_optimizer.minimize(
                        model.disc_loss, var_list=model.W_trans)
    disc_train_step = disc_optimizer.minimize(
                        model.disc_loss, var_list=model.theta_D)

