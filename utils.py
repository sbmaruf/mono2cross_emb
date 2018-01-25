import tensorflow as tf
import numpy as np
import time

def load_emb(src_add, embed_dim, max_vocab, t):
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
            word2id[word] = len(word2id)
            vectors.append(vect[None])
            if i >= max_vocab:
                break

    print("Loaded {0} pre-trained word embeddings".format(len(vectors)))

    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.concatenate(vectors, 0)
    # np.save("./my_emb_"+t, embeddings)
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


def getOptimizer(learning_method, learning_rate, msg=''):
    assert (learning_method >= 0
            and learning_method <= 9
            and type(learning_method) == type(int(1)))
    print(msg+"Using optimizer ", num2optimizer(learning_method))
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
                        model.map_loss, var_list=model.theta_M)
    disc_train_step = disc_optimizer.minimize(
                        model.disc_loss, var_list=model.theta_D)
    return map_train_step, disc_train_step



# def get_minibatch(src_sz, tgt_sz, batch_sz, smooth=.1):
#     cnt = 0
#     while True:
#         src_ids = np.arange(src_sz)
#         tgt_ids = np.arange(tgt_sz)
#         np.random.seed(cnt)
#         np.random.shuffle(src_ids)
#         np.random.shuffle(tgt_ids)
#         idx = np.arange(min(len(src_ids), len(tgt_ids)))
#         idx = idx[::batch_sz]
#         for i in idx:
#             src_x = src_ids[i:i+batch_sz]
#             tgt_x = src_ids[i:i+batch_sz]
#             if len(src_x) != batch_sz or len(tgt_x) != batch_sz:
#                 break
#             src_y = np.vstack([np.tile([1], [batch_sz, 1])])
#             src_y = src_y - smooth
#
#             tgt_y = np.vstack([np.tile([0], [batch_sz, 1])])
#             tgt_y = tgt_y + smooth
#             yield cnt, src_x, src_y, tgt_x, tgt_y
#         cnt+=1

def get_minibatch(batch_sz, frq_x, frq_y):

    while( True ):

        src_x = np.random.randint(low=0, high=frq_x, size=batch_sz)
        tgt_x = np.random.randint(low=0, high=frq_y, size=batch_sz)
        #
        # src_x = np.arange(32)
        # tgt_x = np.arange(32)
        # print(src_x, tgt_x)
        src_y = np.vstack([np.tile([1], [batch_sz, 1])])
        src_y = src_y - .1
        tgt_y = np.vstack([np.tile([0], [batch_sz, 1])])
        tgt_y = tgt_y + .1
        yield src_x, src_y, tgt_x, tgt_y



def save_dump(word2id, id2word, emb, flag):
    np.save("./data/word2id"+flag, word2id)
    np.save("./data/id2word" + flag, id2word)
    np.save("./data/emb" + flag, emb)


def load_dump(flag):
    word2id = np.load("./data/word2id"+flag+".npy").item()
    id2word = np.load("./data/id2word" +flag+".npy").item()
    emb = np.load("./data/emb" + flag + ".npy")
    return word2id, id2word, emb

def save_model( sess, emb_model, src_emb, tgt_emb, src_id2word, tgt_id2word, name ):
    W, _ = sess.run([emb_model.W_trans, emb_model.w_shape])
    src_emb = np.matmul(src_emb, W)
    src_emb = src_emb / np.repeat(np.linalg.norm(src_emb, axis=1, keepdims=True), src_emb.shape[1], axis=1)
    tgt_emb = tgt_emb / np.repeat(np.linalg.norm(tgt_emb, axis=1, keepdims=True), tgt_emb.shape[1], axis=1)
    l = len(src_id2word)
    print("total ", l," source embedding writting on the disk ... .. .")
    with open("./data/"+name+"-src-vec.txt", "w") as f:
        f.write(str(l)+" "+str(src_emb.shape[1])+"\n")
        for i in range(l):
            f.write(str(src_id2word[i])+" "+" ".join(str(x) for x in src_emb[i])+"\n")

    l = len(tgt_id2word)
    print("total ", l, " target embedding writting on the disk ... .. .")
    with open("./data/"+name+"-tgt-vec.txt", "w") as f:
        f.write(str(l) + " " + str(tgt_emb.shape[1])+"\n")
        for i in range(l):
            f.write(str(tgt_id2word[i]) + " " + " ".join(str(x) for x in tgt_emb[i])+"\n")