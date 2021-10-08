# encoding=utf-8
from models import *
from helper import *
from random import *
from pprint import pprint
import pandas as pd
# import networkx as nx
import scipy.sparse as sp
import matplotlib.pyplot as plt, uuid, sys, os, time, argparse
import pickle, pdb, operator, random, sys
import tensorflow as tf
from collections import defaultdict as ddict
from scipy.stats import norm
# from pymongo import MongoClient
from tensorflow.contrib.layers import xavier_initializer as xavier

YEARMIN = -50
YEARMAX = 3000


class TRCHyTE(Model):
    def __init__(self, params):
        print("_init_")
        self.p = params
        self.p.batch_size = self.p.batch_size
        if self.p.l2 == 0.0:
            self.regularizer = None
        else:
            self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.p.l2)
        self.load_data()
        self.nbatches = len(self.data) // self.p.batch_size
        self.add_placeholders()
        self.pos, neg = self.add_model()
        self.loss = self.add_loss(self.pos, neg)
        self.train_op_1 = self.add_optimizer(self.loss)
        self.merged_summ = tf.summary.merge_all()
        self.summ_writer = None
        print('model done')



    def read_valid(self, filename):
        print("read_valid")
        valid_triples = []
        with open(filename, 'r') as filein:
            temp = []
            for line in filein:
                temp = [int(x.strip()) for x in line.split()[0:3]]
                temp.append([line.split()[3], line.split()[4]])
                valid_triples.append(temp)
        return valid_triples

    def getOneHot(self, start_data, end_data, num_class):
        print("getOneHot")
        temp = np.zeros((len(start_data), num_class), np.float32)
        for i, ele in enumerate(start_data):
            if end_data[i] >= start_data[i]:
                temp[i, start_data[i]:end_data[i] + 1] = 1 / (end_data[i] + 1 - start_data[i])
            else:
                pdb.set_trace()
        return temp

    def getBatches(self, data, shuffle=True):
        # print("getBatches")
        if shuffle: random.shuffle(data)
        num_batches = len(data) // self.p.batch_size

        for i in range(num_batches):
            start_idx = i * self.p.batch_size
            yield data[start_idx : start_idx + self.p.batch_size]

    def create_year2id(self, triple_time):
        print("creat_year2id")
        year2id = dict()
        freq = ddict(int)
        count = 0
        year_list = []

        for k, v in triple_time.items():
            try:
                start = v[0].split('-')[0]
                end = v[1].split('-')[0]
            except:
                pdb.set_trace()

            if start.find('#') == -1 and len(start) == 4: year_list.append(int(start))
            if end.find('#') == -1 and len(end) == 4: year_list.append(int(end))

        year_list.sort()
        for year in year_list:
            freq[year] = freq[year] + 1

        year_class = []
        count = 0
        for key in sorted(freq.keys()):
            count += freq[key]
            if count > 300:
                year_class.append(key)
                count = 0
        prev_year = 0
        i = 0
        for i, yr in enumerate(year_class):
            year2id[(prev_year, yr)] = i
            prev_year = yr + 1
        year2id[(prev_year, max(year_list))] = i + 1
        self.year_list = year_list
        return year2id

    def get_span_ids(self, start, end):
        # print("get_span_ids")
        start = int(start)
        end = int(end)
        if start > end:
            end = YEARMAX

        if start == YEARMIN:
            start_lbl = 0
        else:
            for key, lbl in sorted(self.year2id.items(), key=lambda x: x[1]):
                if start >= key[0] and start <= key[1]:
                    start_lbl = lbl

        if end == YEARMAX:
            end_lbl = len(self.year2id.keys()) - 1
        else:
            for key, lbl in sorted(self.year2id.items(), key=lambda x: x[1]):
                if end >= key[0] and end <= key[1]:
                    end_lbl = lbl
        return start_lbl, end_lbl

    def create_id_labels(self, triple_time, dtype):
        def get_quatity(st, et, nt):
            if et - st > 1:
                return norm.cdf(nt - st) - norm.cdf(nt - et)
            else:
                return 1
        print("create_id_labels")
        YEARMAX = 3000
        YEARMIN = -50

        inp_idx, start_idx, end_idx, quatity_idx = [], [], [], []

        for k, v in triple_time.items():
            start = v[0].split('-')[0]
            end = v[1].split('-')[0]
            if start == '####':
                start = YEARMIN
            elif start.find('#') != -1 or len(start) != 4:
                continue

            if end == '####':
                end = YEARMAX
            elif end.find('#') != -1 or len(end) != 4:
                continue

            start = int(start)
            end = int(end)
            if start > end:
                end = YEARMAX
            inp_idx.append(k)
            if start == YEARMIN:
                start_idx.append(0)
                quatity_idx.append(1)
            else:
                for key, lbl in sorted(self.year2id.items(), key=lambda x: x[1]):
                    if start >= key[0] and start <= key[1]:
                        start_idx.append(lbl)
                        quatity_idx.append(0.8+self.p.quatity_factor)
            if end == YEARMAX:
                end_idx.append(len(self.year2id.keys()) - 1)
            else:
                for key, lbl in sorted(self.year2id.items(), key=lambda x: x[1]):
                    if end >= key[0] and end <= key[1]:
                        end_idx.append(lbl)
        return inp_idx, start_idx, end_idx, quatity_idx

    def load_data(self):
        def get_quatity(st, et, nt,scale):
            if st == et:
                return 1
            else:
                return 1 - norm.cdf(nt - et,scale=scale)
        print("load_data")
        r_sd = []
        with open(self.p.SD, 'r', ) as f:
            for line in f.readlines():
                line = line.strip('\n')  # 去掉换行符\n
                b = line.split('\t')  # 将每一行以空格为分隔符转换成列表
                r_sd.append(b)
        r_sd = dict(r_sd)
        triple_set = []
        with open(self.p.triple2id, 'r') as filein:
            for line in filein:
                tup = (int(line.split()[0].strip()), int(line.split()[1].strip()), int(line.split()[2].strip()))
                triple_set.append(tup)
        triple_set = set(triple_set)

        train_triples = []
        self.start_time, self.end_time, self.num_class = ddict(dict), ddict(dict), ddict(dict)
        triple_time, entity_time = dict(), dict()
        self.inp_idx, self.start_idx, self.end_idx, self.quatity_idx, self.labels = ddict(list), ddict(list), ddict(
            list), ddict(list), ddict(list)
        max_ent, max_rel, count = 0, 0, 0

        with open(self.p.dataset, 'r') as filein:
            for line in filein:
                train_triples.append([int(x.strip()) for x in line.split()[0:3]])
                triple_time[count] = [x.split('-')[0] for x in line.split()[3:5]]
                count += 1
        with open(self.p.entity2id, 'r', encoding='utf-8') as filein2:
            for line in filein2:
                max_ent = max_ent + 1

        self.year2id = self.create_year2id(triple_time)
        self.inp_idx['triple'], self.start_idx['triple'], self.end_idx['triple'], self.quatity_idx[
            'triple'] = self.create_id_labels(triple_time, 'triple')

        self.num_class = len(self.year2id.keys())

        keep_idx = set(self.inp_idx['triple'])
        for i in range(len(train_triples) - 1, -1, -1):
            if i not in keep_idx:
                del train_triples[i]

        with open(self.p.relation2id, 'r') as filein3:
            for line in filein3:
                max_rel = max_rel + 1
        index = randint(1, len(train_triples)) - 1

        posh, rela, post = zip(*train_triples)
        head, rel, tail = zip(*train_triples)

        posh = list(posh)
        post = list(post)
        rela = list(rela)

        head = list(head)
        tail = list(tail)
        rel = list(rel)

        for i in range(len(posh)):
            if self.start_idx['triple'][i] < self.end_idx['triple'][i]:
                s_temp = self.start_idx['triple'][i]
                e_temp = self.end_idx['triple'][i]
                q = self.p.quatity_factor
                for j in range(self.start_idx['triple'][i] + 1,
                               self.end_idx['triple'][i] + 1):
                    head.append(posh[i])
                    rel.append(rela[i])
                    tail.append(post[i])
                    self.start_idx['triple'].append(j)
                    if r_sd[str(rela[i])] == '0' or r_sd[str(rela[i])] == '0.0':
                        self.quatity_idx['triple'].append(1)
                    else:
                        scale = float(r_sd[str(rela[i])])
                        self.quatity_idx['triple'].append(get_quatity(s_temp, e_temp, j, scale) + q)

        self.ph, self.pt, self.r, self.nh, self.nt, self.triple_time, self.triple_quatity, self.label = [], [], [], [], [], [], [], []
        for triple in range(len(head)):
            neg_set = set()
            for k in range(self.p.M):
                possible_head = randint(0, max_ent - 1)
                while (possible_head, rel[triple], tail[triple]) in triple_set or (
                possible_head, rel[triple], tail[triple]) in neg_set:
                    possible_head = randint(0, max_ent - 1)
                self.nh.append(possible_head)
                self.nt.append(tail[triple])
                self.r.append(rel[triple])
                self.ph.append(head[triple])
                self.pt.append(tail[triple])
                self.triple_time.append(self.start_idx['triple'][triple])
                self.triple_quatity.append(self.quatity_idx['triple'][triple])
                self.label.append(k)
                neg_set.add((possible_head, rel[triple], tail[triple]))

        for triple in range(len(tail)):
            neg_set = set()
            for k in range(self.p.M):
                possible_tail = randint(0, max_ent - 1)
                while (head[triple], rel[triple], possible_tail) in triple_set or (
                head[triple], rel[triple], possible_tail) in neg_set:
                    possible_tail = randint(0, max_ent - 1)
                self.nh.append(head[triple])
                self.nt.append(possible_tail)
                self.r.append(rel[triple])
                self.ph.append(head[triple])
                self.pt.append(tail[triple])
                self.triple_time.append(self.start_idx['triple'][triple])
                self.triple_quatity.append(self.quatity_idx['triple'][triple])
                self.label.append(k)
                neg_set.add((head[triple], rel[triple], possible_tail))


        self.max_rel = max_rel
        self.max_ent = max_ent
        self.max_time = len(self.year2id.keys())
        self.data = list(zip(self.ph, self.pt, self.r, self.nh, self.nt, self.triple_time, self.triple_quatity, self.label))
        self.data = self.data + self.data[0:self.p.batch_size]

    def add_placeholders(self):
        print("add_placeholders")
        self.start_year = tf.placeholder(tf.int32, shape=[None], name='start_time')
        self.quatity_fac = tf.placeholder(tf.float32, shape=[None,1])
        self.end_year = tf.placeholder(tf.int32, shape=[None], name='end_time')
        self.pos_head = tf.placeholder(tf.int32, [None, 1])
        self.pos_tail = tf.placeholder(tf.int32, [None, 1])
        self.rel = tf.placeholder(tf.int32, [None, 1])
        self.neg_head = tf.placeholder(tf.int32, [None, 1])
        self.neg_tail = tf.placeholder(tf.int32, [None, 1])
        self.mode = tf.placeholder(tf.int32, shape=())
        self.pred_mode = tf.placeholder(tf.int32, shape=())
        self.query_mode = tf.placeholder(tf.int32, shape=())
        self.label_index = tf.placeholder(dtype=tf.int32, shape=[1, None], name='label_index')

    def create_feed_dict(self, batch, wLabels=True, dtype='train'):
        # print("creat_feed_dict")
        ph, pt, r, nh, nt, start_idx, quatity_idx, label_index = zip(*batch)
        feed_dict = {}
        feed_dict[self.pos_head] = np.array(ph).reshape(-1, 1)
        feed_dict[self.pos_tail] = np.array(pt).reshape(-1, 1)
        feed_dict[self.rel] = np.array(r).reshape(-1, 1)
        feed_dict[self.start_year] = np.array(start_idx)
        if dtype == 'train':
            feed_dict[self.neg_head] = np.array(nh).reshape(-1, 1)
            feed_dict[self.neg_tail] = np.array(nt).reshape(-1, 1)
            feed_dict[self.quatity_fac] = np.array(quatity_idx).reshape(-1, 1)
            feed_dict[self.mode] = 1
            feed_dict[self.pred_mode] = 0
            feed_dict[self.query_mode] = 0
            feed_dict[self.label_index] = np.array(label_index).reshape(1, -1)
        else:
            feed_dict[self.mode] = -1

        return feed_dict

    def time_projection(self, data, t):
        print("time_projection")
        inner_prod = tf.tile(tf.expand_dims(tf.reduce_sum(data * t, axis=1), axis=1), [1, self.p.inp_dim])
        prod = t * inner_prod
        data = data - prod
        return data

    def add_model(self):
        def call_train():
            return tf.multiply(q_1, pos)

        def call_test():
            return pos
        print("add_model")
        # nn_in = self.input_x
        with tf.name_scope("embedding"):
            self.ent_embeddings = tf.get_variable(name="ent_embedding", shape=[self.max_ent, self.p.inp_dim],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                                  regularizer=self.regularizer)
            self.rel_embeddings = tf.get_variable(name="rel_embedding", shape=[self.max_rel, self.p.inp_dim],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                                  regularizer=self.regularizer)
            self.time_embeddings = tf.get_variable(name="time_embedding", shape=[self.max_time, self.p.inp_dim],
                                                   initializer=tf.contrib.layers.xavier_initializer(uniform=False))

        transE_in_dim = self.p.inp_dim
        transE_in = self.ent_embeddings


        neutral = tf.constant(0)  ## mode = 1 for train mode = -1 test
        test_type = tf.constant(0)  ##  pred_mode = 1 for head -1 for tail
        query_type = tf.constant(0)  ## query mode  =1 for head tail , -1 for rel

        def f_train():
            pos_h_e = tf.squeeze(tf.nn.embedding_lookup(transE_in, self.pos_head))
            pos_t_e = tf.squeeze(tf.nn.embedding_lookup(transE_in, self.pos_tail))
            pos_r_e = tf.squeeze(tf.nn.embedding_lookup(self.rel_embeddings, self.rel))
            return pos_h_e, pos_t_e, pos_r_e

        def f_test():
            def head_tail_query():
                def f_head():
                    e2 = tf.squeeze(tf.nn.embedding_lookup(transE_in, self.pos_tail))
                    pos_h_e = transE_in
                    pos_t_e = tf.reshape(tf.tile(e2, [self.max_ent]), (self.max_ent, transE_in_dim))
                    return pos_h_e, pos_t_e

                def f_tail():
                    e1 = tf.squeeze(tf.nn.embedding_lookup(transE_in, self.pos_head))
                    pos_h_e = tf.reshape(tf.tile(e1, [self.max_ent]), (self.max_ent, transE_in_dim))
                    pos_t_e = transE_in
                    return pos_h_e, pos_t_e

                pos_h_e, pos_t_e = tf.cond(self.pred_mode > test_type, f_head, f_tail)
                r = tf.squeeze(tf.nn.embedding_lookup(self.rel_embeddings, self.rel))
                pos_r_e = tf.reshape(tf.tile(r, [self.max_ent]), (self.max_ent, transE_in_dim))
                return pos_h_e, pos_t_e, pos_r_e

            def rel_query():
                e1 = tf.squeeze(tf.nn.embedding_lookup(transE_in, self.pos_head))
                e2 = tf.squeeze(tf.nn.embedding_lookup(transE_in, self.pos_tail))
                pos_h_e = tf.reshape(tf.tile(e1, [self.max_rel]), (self.max_rel, transE_in_dim))
                pos_t_e = tf.reshape(tf.tile(e2, [self.max_rel]), (self.max_rel, transE_in_dim))
                pos_r_e = self.rel_embeddings
                return pos_h_e, pos_t_e, pos_r_e

            pos_h_e, pos_t_e, pos_r_e = tf.cond(self.query_mode > query_type, head_tail_query, rel_query)
            return pos_h_e, pos_t_e, pos_r_e

        pos_h_e, pos_t_e, pos_r_e = tf.cond(self.mode > neutral, f_train, f_test)
        neg_h_e = tf.squeeze(tf.nn.embedding_lookup(transE_in, self.neg_head))
        neg_t_e = tf.squeeze(tf.nn.embedding_lookup(transE_in, self.neg_tail))

        #### ----- time -----###
        sy1 = self.start_year
        q_1 = self.quatity_fac
        q_1_print = tf.Print(q_1,['q_1 = ',q_1])


        t_1 = tf.squeeze(tf.nn.embedding_lookup(self.time_embeddings, sy1))


        pos_h_e_t_1 = self.time_projection(pos_h_e, t_1)

        neg_h_e_t_1 = self.time_projection(neg_h_e, t_1)
        pos_t_e_t_1 = self.time_projection(pos_t_e, t_1)
        neg_t_e_t_1 = self.time_projection(neg_t_e, t_1)
        pos_r_e_t_1 = self.time_projection(pos_r_e, t_1)

        with tf.variable_scope("attention", initializer=xavier(), dtype=tf.float32):
            relation_matrixs = []

            init_file = './data/initial_vectors/init_vec'
            init_vec = pickle.load(open(init_file, 'rb'))
            self.layer = (1 + np.max(init_vec['relation_levels'], 0)).astype(np.int32)
            hier = init_vec['relation_levels'].shape[1]
            for i in range(hier):
                relation_matrixs.append(self.GetVar(init_vec=init_vec, key='relmat' + str(i),
                                                     name='relation_matrix_l' + str(i),
                                                     shape=[self.layer[i], self.p.hidden_size]))

            self.relation_levels = tf.constant(init_vec['relation_levels'], shape=[self.max_rel, self.p.hire_shape],
                                               dtype=tf.int32, name='relation_levels')
            label_layer = tf.nn.embedding_lookup(self.relation_levels, self.label_index)
            attention_logits = []
            for i in range(hier):
                current_relation = tf.nn.embedding_lookup(relation_matrixs[i], label_layer[:, 0])
                attention_logits.append(tf.reduce_sum(current_relation, 1))

            attention_logits_stack = tf.stack(attention_logits)
            attention_score_hidden = tf.nn.softmax(attention_logits_stack, 1)
            attention_score_hidden = tf.reshape(attention_score_hidden, [-1])

            stack_repre = tf.layers.dropout(attention_score_hidden, rate=1 - self.p.keep_prob, training=True)
            stack_repre = tf.reduce_sum(stack_repre, 0, keep_dims=True)
        if self.p.L1_flag:
            print("start q_1")
            pos = tf.reduce_sum(abs(pos_h_e_t_1 + pos_r_e_t_1 - pos_t_e_t_1), 1, keep_dims=True)
            pos_print = tf.Print(pos, ['pos = ', pos])
            pos = tf.cond(self.mode > neutral, call_train, call_test)
            pos = tf.multiply(stack_repre, pos)
            print("end q_1")
            neg = tf.reduce_sum(abs(neg_h_e_t_1 + pos_r_e_t_1 - neg_t_e_t_1), 1, keep_dims=True)

        else:
            print("start L2")
            pos = tf.reduce_sum((pos_h_e_t_1 + pos_r_e_t_1 - pos_t_e_t_1) ** 2, 1, keep_dims=True)
            neg = tf.reduce_sum((neg_h_e_t_1 + pos_r_e_t_1 - neg_t_e_t_1) ** 2, 1, keep_dims=True)
        '''
		debug_nn([self.pred_mode,self.mode], feed_dict = self.create_feed_dict(self.data[0:self.p.batch_size],dtype='test'))
		'''
        return pos, neg

    def GetVar(self, init_vec, key, name, shape=None, initializer=None, trainable=True):

        if init_vec is not None and key in init_vec:
            return tf.get_variable(name=name, initializer=init_vec[key], trainable=trainable)
        else:
            return tf.get_variable(name=name, shape=shape, initializer=initializer, trainable=trainable)

    def add_loss(self, pos, neg,):
        print("add_loss")
        with tf.name_scope('Loss_op'):
            loss = tf.reduce_sum(tf.maximum(pos - neg + self.p.margin, 0))
            if self.regularizer != None: loss += tf.contrib.layers.apply_regularization(self.regularizer,
                                                                                        tf.get_collection(
                                                                                            tf.GraphKeys.REGULARIZATION_LOSSES))
            return loss


    def add_optimizer(self, loss):
        print("add_optimizer")
        with tf.name_scope('Optimizer'):
            optimizer_1 = tf.train.AdamOptimizer(self.p.lr)
            train_op_1 = optimizer_1.minimize(loss)

        time_normalizer = tf.assign(self.time_embeddings, tf.nn.l2_normalize(self.time_embeddings, dim=1))
        return train_op_1

    def run_epoch(self, sess, data, epoch):
        losses = []
        for step, batch in enumerate(self.getBatches(data, shuffle)):
            feed = self.create_feed_dict(batch)
            l, a = sess.run([self.loss, self.train_op_1], feed_dict=feed)
            losses.append(l)
        return np.mean(losses)

    def fit(self, sess):
        print("fit")
        saver = tf.train.Saver(max_to_keep=None)
        self.p.name = self.p.name.replace(':', '', 3)
        save_dir = 'checkpoints/' + self.p.name + '/'
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        save_dir_results = './results/' + self.p.name + '/'
        if not os.path.exists(save_dir_results): os.makedirs(save_dir_results)
        if self.p.restore:
            save_path = os.path.join(save_dir, 'epoch_{}'.format(self.p.restore_epoch))
            saver.restore(sess, save_path)

        print('start fitting')
        validation_data = self.read_valid(self.p.test_data)
        for epoch in range(self.p.max_epochs):
            l = self.run_epoch(sess, self.data, epoch)
            if epoch % 5 == 0:
                print('Epoch {}\tLoss {}\t model {}'.format(epoch, l, self.p.name))
            if epoch % self.p.test_freq == 0 and epoch != 0:
                save_path = os.path.join(save_dir, 'epoch_{}'.format(epoch))  ## -- check pointing -- ##
                saver.save(sess=sess, save_path=save_path)
                if epoch == self.p.test_freq:
                    f_valid = open(save_dir_results + '/valid.txt', 'w')

                fileout_head = open(save_dir_results + '/valid_head_pred_{}.txt'.format(epoch), 'w')
                fileout_tail = open(save_dir_results + '/valid_tail_pred_{}.txt'.format(epoch), 'w')
                fileout_rel = open(save_dir_results + '/valid_rel_pred_{}.txt'.format(epoch), 'w')

                for i, t in enumerate(validation_data):
                    loss = np.zeros(self.max_ent)
                    start_trip = t[3][0].split('-')[0]
                    end_trip = t[3][1].split('-')[0]
                    if start_trip == '####':
                        start_trip = YEARMIN
                    elif start_trip.find('#') != -1 or len(start_trip) != 4:
                        continue

                    if end_trip == '####':
                        end_trip = YEARMAX
                    elif end_trip.find('#') != -1 or len(end_trip) != 4:
                        continue

                    start_lbl, end_lbl = self.get_span_ids(start_trip, end_trip)
                    if epoch == self.p.test_freq:
                        f_valid.write(str(t[0]) + '\t' + str(t[1]) + '\t' + str(t[2]) + '\n')
                    pos_head = sess.run(self.pos, feed_dict={self.pos_head: np.array([t[0]]).reshape(-1, 1),
                                                   self.rel: np.array([t[1]]).reshape(-1, 1),
                                                   self.pos_tail: np.array([t[2]]).reshape(-1, 1),
                                                   self.start_year: np.array([start_lbl] * self.max_ent),
                                                   self.end_year: np.array([end_lbl] * self.max_ent),
                                                   self.quatity_fac: np.array([[1]]).reshape(-1, 1),
                                                   self.label_index: np.array([[1]]).reshape(1, -1),
                                                   self.mode: -1,
                                                   self.pred_mode: 1,
                                                   self.query_mode: 1})
                    pos_head = np.squeeze(pos_head)

                    pos_tail = sess.run(self.pos, feed_dict={self.pos_head: np.array([t[0]]).reshape(-1, 1),
                                                             self.rel: np.array([t[1]]).reshape(-1, 1),
                                                             self.pos_tail: np.array([t[2]]).reshape(-1, 1),
                                                             self.start_year: np.array([start_lbl] * self.max_ent),
                                                             self.quatity_fac: np.array([[1]]).reshape(-1, 1),
                                                             self.label_index: np.array([[1]]).reshape(1, -1),
                                                             self.end_year: np.array([end_lbl] * self.max_ent),
                                                             self.mode: -1,
                                                             self.pred_mode: -1,
                                                             self.query_mode: 1})
                    pos_tail = np.squeeze(pos_tail)

                    pos_rel = sess.run(self.pos, feed_dict={self.pos_head: np.array([t[0]]).reshape(-1, 1),
                                                            self.rel: np.array([t[1]]).reshape(-1, 1),
                                                            self.pos_tail: np.array([t[2]]).reshape(-1, 1),
                                                            self.start_year: np.array([start_lbl] * self.max_rel),
                                                            self.end_year: np.array([end_lbl] * self.max_rel),
                                                            self.quatity_fac: np.array([[1]]).reshape(-1, 1),
                                                            self.label_index: np.array([[1]]).reshape(1, -1),
                                                            self.mode: -1,
                                                            self.pred_mode: -1,
                                                            self.query_mode: -1})
                    pos_rel = np.squeeze(pos_rel)
                    fileout_head.write(' '.join([str(x) for x in pos_head]) + '\n')
                    fileout_tail.write(' '.join([str(x) for x in pos_tail]) + '\n')
                    fileout_rel.write(' '.join([str(x) for x in pos_rel]) + '\n')

                    if i % 500 == 0:
                        print('{}. no of valid_triples complete'.format(i))
                # if i%4000 == 0 and i!=0: break
                fileout_head.close()
                fileout_tail.close()
                fileout_rel.close()
                if epoch == self.p.test_freq:
                    f_valid.close()
                print("Validation Ended")


if __name__ == "__main__":
    print('here in main')
    parser = argparse.ArgumentParser(description='TRCHyTE')

    parser.add_argument('-data_type', dest="data_type", default='yago', choices=['yago', 'wiki_data','WDP','YGP'],
                        help='dataset to choose')
    parser.add_argument('-version', dest='version', default='large', choices=['large', 'small'],
                        help='data version to choose')
    parser.add_argument('-test_freq', dest="test_freq", default=20, type=int, help='Batch size')
    parser.add_argument('-neg_sample', dest="M", default=10, type=int, help='Batch size')
    parser.add_argument('-gpu', dest="gpu", default='1', help='GPU to use')
    parser.add_argument('-drop', dest="dropout", default=1.0, type=float, help='Dropout for full connected layer')
    parser.add_argument('-rdrop', dest="rec_dropout", default=1.0, type=float, help='Recurrent dropout for LSTM')
    parser.add_argument('-lr', dest="lr", default=0.0001, type=float, help='Learning rate')
    parser.add_argument('-lam_1', dest="lambda_1", default=0.4, type=float, help='transE weight')
    parser.add_argument('-lam_2', dest="lambda_2", default=0.25, type=float, help='entitty loss weight')
    parser.add_argument('-margin', dest="margin", default=10, type=float, help='margin')
    parser.add_argument('-batch', dest="batch_size", default=50000, type=int, help='Batch size')  # 50000
    parser.add_argument('-epoch', dest="max_epochs", default=1000, type=int, help='Max epochs')  # 1000
    parser.add_argument('-l2', dest="l2", default=0.0, type=float, help='L2 regularization')
    parser.add_argument('-seed', dest="seed", default=1234, type=int, help='Seed for randomization')
    parser.add_argument('-inp_dim', dest="inp_dim", default=128, type=int, help='Hidden state dimension of Bi-LSTM')
    parser.add_argument('-L1_flag', dest="L1_flag", action='store_false', help='Hidden state dimension of FC layer')
    parser.add_argument('-onlytransE', dest="onlytransE", action='store_true',
                        help='Evaluate model on only transE loss')
    parser.add_argument('-restore', dest="restore", action='store_true',
                        help='Restore from the previous best saved model')
    parser.add_argument('-res_epoch', dest="restore_epoch", default=200, type=int,
                        help='Restore from the previous best saved model')
    parser.add_argument('-name', dest="name", default='test', help='Name of the run')
    parser.add_argument('-quatity_factor', dest="quatity_factor", default=0.6, type=float, help='q')
    parser.add_argument('-hidden_size', dest="hidden_size",default=106, type=int, help="hidden_size")
    parser.add_argument('-weight_decay', dest="weight_decay", default=0.00001, type=float, help="weight_decay")
    parser.add_argument('-keep_prob', dest="keep_prob", default=0.2, type=float, help="keep_prob")
    parser.add_argument('-hire_shape', dest="keep_prob", default=106, type=int, help="hire_shape")
    args = parser.parse_args()
    args.dataset = 'data/' + args.data_type + '/' + args.version + '/train.txt'
    args.SD = 'data/' + args.data_type + '/' + args.version + '/SD.txt'
    args.entity2id = 'data/' + args.data_type + '/' + args.version + '/entity2id.txt'
    args.relation2id = 'data/' + args.data_type + '/' + args.version + '/relation2id.txt'
    args.test_data = 'data/' + args.data_type + '/' + args.version + '/valid.txt'
    args.triple2id = 'data/' + args.data_type + '/' + args.version + '/triple2id.txt'
    if not args.restore: args.name = args.name + '_' + time.strftime("%d_%m_%Y") + '_' + time.strftime("%H:%M:%S")
    tf.set_random_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    set_gpu(args.gpu)
    model = TRCHyTE(args)
    print('model object created')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        print('enter fitting')
        model.fit(sess)
