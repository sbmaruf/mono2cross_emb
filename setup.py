import os
from utils import load_emb, \
    getOptimizer, \
    grad_compute
from model import *
import time




src_add = "./data/sskip.100.vectors"
tgt_add = "./data/ES64"
embed_dim = 100

assert os.path.isfile(src_add)
assert os.path.isfile(tgt_add)

src_word2id, src_id2word, src_emb = load_emb(src_add, embed_dim)
tgt_word2id, tgt_id2word, tgt_emb = load_emb(tgt_add, embed_dim)
print(src_emb.shape, tgt_emb.shape)

num_of_epoch = 5
num_of_batch = 1000000
batch_size = 32


emb_model = model(src_emb, tgt_emb)

disc_optimizer = getOptimizer(0, .1)
map_optimizer = getOptimizer(0, .1)
train_step = grad_compute(map_optimizer, disc_optimizer)


# for i_epoch in range(num_of_epoch):
#
#     print("Starting adversarial training epoch {} ... .. .".format(i_epoch))
#     tic = time.time()
#     stats = {'DIS_COSTS': []}
#
#     for n_iter in range(0, num_of_epoch):
#
#         # discriminator training
#         for _ in range(params.dis_steps):
#             trainer.dis_step(stats)
#
#         # mapping training (discriminator fooling)
#         n_words_proc += trainer.mapping_step(stats)
#
#         # log stats
#         if n_iter % 500 == 0:
#             stats_str = [('DIS_COSTS', 'Discriminator loss')]
#             stats_log = ['%s: %.4f' % (v, np.mean(stats[k]))
#                          for k, v in stats_str if len(stats[k]) > 0]
#             stats_log.append('%i samples/s' % int(n_words_proc / (time.time() - tic)))
#             logger.info(('%06i - ' % n_iter) + ' - '.join(stats_log))
#
#             # reset
#             tic = time.time()
#             n_words_proc = 0
#             for k, _ in stats_str:
#                 del stats[k][:]
#
#     # embeddings / discriminator evaluation
#     to_log = OrderedDict({'n_epoch': n_epoch})
#     evaluator.all_eval(to_log)
#     evaluator.eval_dis(to_log)
#
#     # JSON log / save best model / end of epoch
#     logger.info("__log__:%s" % json.dumps(to_log))
#     trainer.save_best(to_log, VALIDATION_METRIC)
#     logger.info('End of epoch %i.\n\n' % n_epoch)
#
#     # update the learning rate (stop if too small)
#     trainer.update_lr(to_log, VALIDATION_METRIC)
#     if trainer.map_optimizer.param_groups[0]['lr'] < params.min_lr:
#         logger.info('Learning rate < 1e-6. BREAK.')
#         break
