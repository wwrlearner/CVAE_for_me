from __future__ import print_function
import numpy as np
import logging


class BioDataLoader(object):
    logger = logging.getLogger()

    def __init__(self, name, fix_batch=True):
        self.batch_size = 0
        self.ptr = 0
        self.num_batch = None
        self.indexes = None
        self.data_size = None
        self.batch_indexes = None
        self.fix_batch = fix_batch
        self.max_seq_len = None
        self.name = name

    def _shuffle_indexes(self):
        np.random.shuffle(self.indexes)

    def _shuffle_batch_indexes(self):
        np.random.shuffle(self.batch_indexes)

    def _prepare_batch(self, *args, **kwargs):
        raise NotImplementedError("Have to override prepare batch")

    def epoch_init(self, config, shuffle=True, verbose=True):
        self.ptr = 0
        self.batch_size = config.batch_size
        self.num_batch = self.data_size // config.batch_size
        if verbose:
            self.logger.info("Number of left over samples %d" % (self.data_size - config.batch_size * self.num_batch))

        if shuffle and not self.fix_batch:
            self._shuffle_indexes()

        self.batch_indexes = []
        for i in range(self.num_batch):
            self.batch_indexes.append(self.indexes[i * self.batch_size:(i + 1) * self.batch_size])

        if shuffle and self.fix_batch:
            self._shuffle_batch_indexes()

        if verbose:
            self.logger.info("%s begins with %d batches" % (self.name, self.num_batch))

    def next_batch(self):
        if self.ptr < self.num_batch:
            selected_ids = self.batch_indexes[self.ptr]
            self.ptr += 1
            return self._prepare_batch(selected_index=selected_ids)
        else:
            return None