from __future__ import print_function
import numpy as np
import logging


class DataLoader(object):
    logger = logging.getLogger()

    def __init__(self, name, fix_batch=True): # fix_batch：布尔值，表示是否对每个 batch 固定顺序（默认为 True）。
        self.batch_size = 0
        self.ptr = 0
        self.num_batch = None
        self.indexes = None
        self.data_size = None
        self.batch_indexes = None
        self.fix_batch=fix_batch
        self.max_utt_size = None
        self.name = name # 数据加载器的名称，便于标识

    # 用于打乱 indexes，随机化数据的顺序。
    def _shuffle_indexes(self): 
        np.random.shuffle(self.indexes)

    # 用于打乱 batch_indexes，随机化批次的顺序。
    def _shuffle_batch_indexes(self):
        np.random.shuffle(self.batch_indexes)

    # 抽象方法，用于准备一个批次的数据，必须在继承类中实现。
    def _prepare_batch(self, *args, **kwargs):
        raise NotImplementedError("Have to override prepare batch")

    # 初始化一个训练轮次（epoch），设置批次大小和数据顺序。
    def epoch_init(self, config, shuffle=True, verbose=True):
        self.ptr = 0
        self.batch_size = config.batch_size
        self.num_batch = self.data_size // config.batch_size
        if verbose:
            self.logger.info("Number of left over sample %d" % (self.data_size - config.batch_size * self.num_batch))

        # if shuffle and we want to group lines, shuffle batch indexes
        if shuffle and not self.fix_batch:
            self._shuffle_indexes()

        self.batch_indexes = []
        for i in range(self.num_batch):
            self.batch_indexes.append(self.indexes[i * self.batch_size:(i + 1) * self.batch_size])

        if shuffle and self.fix_batch:
            self._shuffle_batch_indexes()

        if verbose:
            self.logger.info("%s begins with %d batches" % (self.name, self.num_batch))

    # 用于获取下一个批次的数据。
    def next_batch(self):
        if self.ptr < self.num_batch:
            selected_ids = self.batch_indexes[self.ptr]
            self.ptr += 1
            return self._prepare_batch(selected_index=selected_ids)
        else:
            return None


    def pad_to(self, max_len, tokens, do_pad=True):
        if len(tokens) >= max_len:
            return tokens[0:max_len - 1] + [tokens[-1]]
        elif do_pad:
            return tokens + [0] * (max_len - len(tokens))
        else:
            return tokens





