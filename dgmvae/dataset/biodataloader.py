import logging
import numpy as np

class DataLoader(object):
    logger = logging.getLogger()

    def __init__(self, name, config, data, fix_batch=True):
        # data: 字典形式的数据集，包括 'x', 'u', 'z'
        self.name = name
        self.data = data
        self.batch_size = 0
        self.ptr = 0
        # self.num_batch = config.num_batch
        self.indexes = None
        self.data_size = len(data['x'])
        self.batch_indexes = None
        self.fix_batch = fix_batch

    # 用于打乱 indexes，随机化数据的顺序。
    def _shuffle_indexes(self):
        np.random.shuffle(self.indexes)

    # 用于打乱 batch_indexes，随机化批次的顺序。
    def _shuffle_batch_indexes(self):
        np.random.shuffle(self.batch_indexes)

    # 初始化一个训练轮次（epoch），设置批次大小和数据顺序。
    def epoch_init(self, config, shuffle=True):
        self.ptr = 0
        self.batch_size = config.batch_size
        self.num_batch = self.data_size // self.batch_size
        self.indexes = np.arange(self.data_size)

        if shuffle and not self.fix_batch:
            self._shuffle_indexes()

        self.batch_indexes = []
        for i in range(self.num_batch):
            self.batch_indexes.append(self.indexes[i * self.batch_size:(i + 1) * self.batch_size])

        if shuffle and self.fix_batch:
            self._shuffle_batch_indexes()

    # 用于准备一个批次的数据。
    def _prepare_batch(self, selected_index):
        x_batch = self.data['x'][selected_index]
        u_batch = self.data['u'][selected_index]
        z_batch = self.data['z'][selected_index]
        return {'x': x_batch, 'u': u_batch, 'z': z_batch}

    # 用于获取下一个批次的数据。
    def next_batch(self):
        if self.ptr < self.num_batch:
            selected_ids = self.batch_indexes[self.ptr]
            self.ptr += 1
            return self._prepare_batch(selected_index=selected_ids)
        else:
            return None


