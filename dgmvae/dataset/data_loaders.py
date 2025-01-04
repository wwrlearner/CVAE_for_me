#    Copyright (C) 2017 Tiancheng Zhao, Carnegie Mellon University
from __future__ import print_function
import numpy as np
from dgmvae.utils import Pack
from dgmvae.dataset.dataloader_bases import DataLoader


# Stanford Multi Domain
class SMDDataLoader(DataLoader):
    def __init__(self, name, data, config):
        super(SMDDataLoader, self).__init__(name, fix_batch=config.fix_batch)
        self.name = name
        self.max_utt_size = config.max_utt_len
        self.data = self.flatten_dialog(data, config.backward_size)
        self.data_size = len(self.data)
        if config.fix_batch:
            all_ctx_lens = [len(d.context) for d in self.data]
            self.indexes = list(np.argsort(all_ctx_lens))[::-1]
        else:
            self.indexes = list(range(len(self.data)))

    def flatten_dialog(self, data, backward_size):
        results = []
        for dialog in data:
            for i in range(1, len(dialog)):
                e_id = i
                s_id = max(0, e_id - backward_size)
                response = dialog[i].copy()
                # response['utt_orisent'] = response.utt
                response['utt'] = self.pad_to(self.max_utt_size, response.utt, do_pad=False)
                contexts = []
                for turn in dialog[s_id:e_id]:
                    turn['utt'] = self.pad_to(self.max_utt_size, turn.utt, do_pad=False)
                    contexts.append(turn)
                results.append(Pack(context=contexts, response=response))
        return results

    def _prepare_batch(self, selected_index):
        rows = [self.data[idx] for idx in selected_index]
        # input_context, context_lens, floors, topics, a_profiles, b_Profiles, outputs, output_lens
        context_lens, context_utts, out_utts, out_lens = [], [], [], []
        metas = []
        for row in rows:
            ctx = row.context
            resp = row.response

            out_utt = resp.utt
            context_lens.append(len(ctx))
            context_utts.append([turn.utt for turn in ctx])

            out_utt = out_utt
            out_utts.append(out_utt)
            out_lens.append(len(out_utt))
            metas.append(resp.meta)
            # ori_out_utts.append(resp.utt_orisent)

        vec_context_lens = np.array(context_lens)
        vec_context = np.zeros((self.batch_size, np.max(vec_context_lens),
                                self.max_utt_size), dtype=np.int32)
        vec_outs = np.zeros((self.batch_size, np.max(out_lens)), dtype=np.int32)
        vec_out_lens = np.array(out_lens)

        for b_id in range(self.batch_size):
            vec_outs[b_id, 0:vec_out_lens[b_id]] = out_utts[b_id]
            # fill the context tensor
            new_array = np.empty((vec_context_lens[b_id], self.max_utt_size))
            new_array.fill(0)
            for i, row in enumerate(context_utts[b_id]):
                for j, ele in enumerate(row):
                    new_array[i, j] = ele
            vec_context[b_id, 0:vec_context_lens[b_id], :] = new_array

        return Pack(contexts=vec_context, context_lens=vec_context_lens,
                    outputs=vec_outs, output_lens=vec_out_lens,
                    metas=metas)

class SMDDialogSkipLoader(DataLoader):
    def __init__(self, name, data, config):
        super(SMDDialogSkipLoader, self).__init__(name, fix_batch=config.fix_batch)
        self.name = name
        self.max_utt_size = config.max_utt_len
        self.data = self.flatten_dialog(data, config.backward_size)
        self.data_size = len(self.data)
        if config.fix_batch:
            all_ctx_lens = [len(d.context) for d in self.data]
            self.indexes = list(np.argsort(all_ctx_lens))[::-1]
        else:
            self.indexes = list(range(len(self.data)))
        self.source_type = "context"

    def flatten_dialog(self, data, backward_size):
        results = []
        for dialog in data:
            for i in range(1, len(dialog)-1):
                e_id = i
                s_id = max(0, e_id - backward_size)

                response = dialog[i]
                prev = dialog[i - 1]
                next = dialog[i + 1]

                response['utt'] = self.pad_to(self.max_utt_size, response.utt, do_pad=False)
                prev['utt'] = self.pad_to(self.max_utt_size, prev.utt, do_pad=False)
                next['utt'] = self.pad_to(self.max_utt_size, next.utt, do_pad=False)

                contexts = []
                for turn in dialog[s_id:e_id]:
                    turn['utt'] = self.pad_to(self.max_utt_size, turn.utt, do_pad=False)
                    contexts.append(turn)

                results.append(Pack(context=contexts, response=response,
                                    prev_resp=prev, next_resp=next))
        return results

    def _prepare_batch(self, selected_index):
        rows = [self.data[idx] for idx in selected_index]

        context_lens, context_utts, out_utts, out_lens = [], [], [], []
        prev_utts, prev_lens = [], []
        next_utts, next_lens = [], []
        metas = []
        for row in rows:
            ctx = row.context
            resp = row.response

            out_utt = resp.utt
            context_lens.append(len(ctx))
            context_utts.append([turn.utt for turn in ctx])

            out_utt = out_utt
            out_utts.append(out_utt)
            out_lens.append(len(out_utt))
            metas.append(resp.meta)

            prev_utts.append(row.prev_resp.utt)
            prev_lens.append(len(row.prev_resp.utt))

            next_utts.append(row.next_resp.utt)
            next_lens.append(len(row.next_resp.utt))

        vec_context_lens = np.array(context_lens)
        vec_context = np.zeros((self.batch_size, np.max(vec_context_lens),
                                self.max_utt_size), dtype=np.int32)
        vec_outs = np.zeros((self.batch_size, np.max(out_lens)), dtype=np.int32)
        vec_prevs = np.zeros((self.batch_size, np.max(prev_lens)), dtype=np.int32)
        vec_nexts = np.zeros((self.batch_size, np.max(next_lens)),dtype=np.int32)
        vec_out_lens = np.array(out_lens)
        vec_prev_lens = np.array(prev_lens)
        vec_next_lens = np.array(next_lens)

        for b_id in range(self.batch_size):
            vec_outs[b_id, 0:vec_out_lens[b_id]] = out_utts[b_id]
            vec_prevs[b_id, 0:vec_prev_lens[b_id]] = prev_utts[b_id]
            vec_nexts[b_id, 0:vec_next_lens[b_id]] = next_utts[b_id]

            # fill the context tensor
            new_array = np.empty((vec_context_lens[b_id], self.max_utt_size))
            new_array.fill(0)
            for i, row in enumerate(context_utts[b_id]):
                for j, ele in enumerate(row):
                    new_array[i, j] = ele
            vec_context[b_id, 0:vec_context_lens[b_id], :] = new_array

        return Pack(contexts=vec_context, context_lens=vec_context_lens,
                    outputs=vec_outs, output_lens=vec_out_lens,
                    metas=metas, prevs=vec_prevs, prev_lens=vec_prev_lens,
                    nexts=vec_nexts, next_lens=vec_next_lens)

# Daily Dialog
class DailyDialogSkipLoader(DataLoader):
    def __init__(self, name, data, config):
        super(DailyDialogSkipLoader, self).__init__(name, fix_batch=config.fix_batch)
        self.name = name
        self.max_utt_size = config.max_utt_len
        self.data = self.flatten_dialog(data, config.backward_size)
        self.data_size = len(self.data)
        if config.fix_batch:
            all_ctx_lens = [len(d.context) for d in self.data]
            self.indexes = list(np.argsort(all_ctx_lens))[::-1]
        else:
            self.indexes = list(range(len(self.data)))

    def flatten_dialog(self, data, backward_size):
        results = []
        for dialog in data:
            for i in range(1, len(dialog)-1):
                e_id = i
                s_id = max(0, e_id - backward_size)

                response = dialog[i]
                prev = dialog[i - 1]
                next = dialog[i + 1]

                response['utt'] = self.pad_to(self.max_utt_size,response.utt, do_pad=False)
                prev['utt'] = self.pad_to(self.max_utt_size, prev.utt, do_pad=False)
                next['utt'] = self.pad_to(self.max_utt_size, next.utt, do_pad=False)

                contexts = []
                for turn in dialog[s_id:e_id]:
                    turn['utt'] = self.pad_to(self.max_utt_size, turn.utt, do_pad=False)
                    contexts.append(turn)

                results.append(Pack(context=contexts, response=response,
                                    prev_resp=prev, next_resp=next))
        return results

    def _prepare_batch(self, selected_index):
        rows = [self.data[idx] for idx in selected_index]

        context_lens, context_utts, out_utts, out_lens = [], [], [], []
        prev_utts, prev_lens = [], []
        next_utts, next_lens = [], []
        metas = []
        for row in rows:
            ctx = row.context
            resp = row.response

            out_utt = resp.utt
            context_lens.append(len(ctx))
            context_utts.append([turn.utt for turn in ctx])

            out_utt = out_utt
            out_utts.append(out_utt)
            out_lens.append(len(out_utt))
            metas.append(resp.meta)

            prev_utts.append(row.prev_resp.utt)
            prev_lens.append(len(row.prev_resp.utt))

            next_utts.append(row.next_resp.utt)
            next_lens.append(len(row.next_resp.utt))

        vec_context_lens = np.array(context_lens)
        vec_context = np.zeros((self.batch_size, np.max(vec_context_lens),
                                self.max_utt_size), dtype=np.int32)
        vec_outs = np.zeros((self.batch_size, np.max(out_lens)), dtype=np.int32)
        vec_prevs = np.zeros((self.batch_size, np.max(prev_lens)), dtype=np.int32)
        vec_nexts = np.zeros((self.batch_size, np.max(next_lens)),dtype=np.int32)
        vec_out_lens = np.array(out_lens)
        vec_prev_lens = np.array(prev_lens)
        vec_next_lens = np.array(next_lens)

        for b_id in range(self.batch_size):
            vec_outs[b_id, 0:vec_out_lens[b_id]] = out_utts[b_id]
            vec_prevs[b_id, 0:vec_prev_lens[b_id]] = prev_utts[b_id]
            vec_nexts[b_id, 0:vec_next_lens[b_id]] = next_utts[b_id]

            # fill the context tensor
            new_array = np.empty((vec_context_lens[b_id], self.max_utt_size))
            new_array.fill(0)
            for i, row in enumerate(context_utts[b_id]):
                for j, ele in enumerate(row):
                    new_array[i, j] = ele
            vec_context[b_id, 0:vec_context_lens[b_id], :] = new_array

        return Pack(contexts=vec_context, context_lens=vec_context_lens,
                    outputs=vec_outs, output_lens=vec_out_lens,
                    metas=metas, prevs=vec_prevs, prev_lens=vec_prev_lens,
                    nexts=vec_nexts, next_lens=vec_next_lens)

class DailyDialogSkipLoaderLabel(DataLoader):
    def __init__(self, name, data, config):
        super(DailyDialogSkipLoaderLabel, self).__init__(name, fix_batch=config.fix_batch)
        self.name = name
        self.max_utt_size = config.max_utt_len
        self.data = self.flatten_dialog(data, config.backward_size)
        self.data_size = len(self.data)
        if config.fix_batch:
            all_ctx_lens = [len(d.context) for d in self.data]
            self.indexes = list(np.argsort(all_ctx_lens))[::-1]
        else:
            self.indexes = list(range(len(self.data)))

    def flatten_dialog(self, data, backward_size):
        results = []
        for dialog in data:
            for i in range(1, len(dialog)-1):
                e_id = i
                s_id = max(0, e_id - backward_size)

                response = dialog[i]
                prev = dialog[i - 1]
                next = dialog[i + 1]

                response['utt'] = self.pad_to(self.max_utt_size,response.utt, do_pad=False)
                prev['utt'] = self.pad_to(self.max_utt_size, prev.utt, do_pad=False)
                next['utt'] = self.pad_to(self.max_utt_size, next.utt, do_pad=False)

                contexts = []
                for turn in dialog[s_id:e_id]:
                    turn['utt'] = self.pad_to(self.max_utt_size, turn.utt, do_pad=False)
                    contexts.append(turn)

                results.append(Pack(context=contexts, response=response,
                                    prev_resp=prev, next_resp=next))
        return results

    def _prepare_batch(self, selected_index):
        rows = [self.data[idx] for idx in selected_index]

        context_lens, context_utts, out_utts, out_lens = [], [], [], []
        prev_utts, prev_lens = [], []
        next_utts, next_lens = [], []
        metas = []
        for row in rows:
            ctx = row.context
            resp = row.response

            out_utt = resp.utt
            context_lens.append(len(ctx))
            context_utts.append([turn.utt for turn in ctx])

            out_utt = out_utt
            out_utts.append(out_utt)
            out_lens.append(len(out_utt))
            metas.append(resp.meta)

            prev_utts.append(row.prev_resp.utt)
            prev_lens.append(len(row.prev_resp.utt))

            next_utts.append(row.next_resp.utt)
            next_lens.append(len(row.next_resp.utt))

        vec_context_lens = np.array(context_lens)
        vec_context = np.zeros((self.batch_size, np.max(vec_context_lens),
                                self.max_utt_size), dtype=np.int32)
        vec_outs = np.zeros((self.batch_size, np.max(out_lens)), dtype=np.int32)
        vec_prevs = np.zeros((self.batch_size, np.max(prev_lens)), dtype=np.int32)
        vec_nexts = np.zeros((self.batch_size, np.max(next_lens)),dtype=np.int32)
        vec_out_lens = np.array(out_lens)
        vec_prev_lens = np.array(prev_lens)
        vec_next_lens = np.array(next_lens)

        for b_id in range(self.batch_size):
            vec_outs[b_id, 0:vec_out_lens[b_id]] = out_utts[b_id]
            vec_prevs[b_id, 0:vec_prev_lens[b_id]] = prev_utts[b_id]
            vec_nexts[b_id, 0:vec_next_lens[b_id]] = next_utts[b_id]

            # fill the context tensor
            new_array = np.empty((vec_context_lens[b_id], self.max_utt_size))
            new_array.fill(0)
            for i, row in enumerate(context_utts[b_id]):
                for j, ele in enumerate(row):
                    new_array[i, j] = ele
            vec_context[b_id, 0:vec_context_lens[b_id], :] = new_array


        z_labels = np.zeros((self.batch_size, 2), dtype=np.int32)
        for b_id in range(self.batch_size):
            z_labels[b_id][0] = int(metas[b_id]["emotion"])
            z_labels[b_id][1] = int(metas[b_id]["act"])

        return Pack(contexts=vec_context, context_lens=vec_context_lens,
                    outputs=vec_outs, output_lens=vec_out_lens,
                    metas=metas, prevs=vec_prevs, prev_lens=vec_prev_lens,
                    nexts=vec_nexts, next_lens=vec_next_lens,
                    z_labels=z_labels)

class DailyDialogDataLoader(DataLoader):
    def __init__(self, name, data, config):
        super(DailyDialogDataLoader, self).__init__(name, fix_batch=config.fix_batch)
        self.name = name
        self.max_utt_size = config.max_utt_len
        self.data = self.flatten_dialog(data, config.backward_size)
        self.data_size = len(self.data)
        if config.fix_batch:
            all_ctx_lens = [len(d.context) for d in self.data]
            self.indexes = list(np.argsort(all_ctx_lens))[::-1]
        else:
            self.indexes = list(range(len(self.data)))

    def flatten_dialog(self, data, backward_size):
        results = []
        for dialog in data:
            for i in range(1, len(dialog)):
                e_id = i
                s_id = max(0, e_id - backward_size)
                response = dialog[i].copy()
                # response['utt_orisent'] = response.utt
                response['utt'] = self.pad_to(self.max_utt_size, response.utt, do_pad=False)
                contexts = []
                for turn in dialog[s_id:e_id]:
                    turn['utt'] = self.pad_to(self.max_utt_size, turn.utt, do_pad=False)
                    contexts.append(turn)
                results.append(Pack(context=contexts, response=response))
        return results

    def _prepare_batch(self, selected_index):
        rows = [self.data[idx] for idx in selected_index]
        # input_context, context_lens, floors, topics, a_profiles, b_Profiles, outputs, output_lens
        context_lens, context_utts, out_utts, out_lens = [], [], [], []
        metas = []
        for row in rows:
            ctx = row.context
            resp = row.response

            out_utt = resp.utt
            context_lens.append(len(ctx))
            context_utts.append([turn.utt for turn in ctx])

            out_utt = out_utt
            out_utts.append(out_utt)
            out_lens.append(len(out_utt))
            metas.append(resp.meta)
            # ori_out_utts.append(resp.utt_orisent)

        vec_context_lens = np.array(context_lens)
        vec_context = np.zeros((self.batch_size, np.max(vec_context_lens),
                                self.max_utt_size), dtype=np.int32)
        vec_outs = np.zeros((self.batch_size, np.max(out_lens)), dtype=np.int32)
        vec_out_lens = np.array(out_lens)

        for b_id in range(self.batch_size):
            vec_outs[b_id, 0:vec_out_lens[b_id]] = out_utts[b_id]
            # fill the context tensor
            new_array = np.empty((vec_context_lens[b_id], self.max_utt_size))
            new_array.fill(0)
            for i, row in enumerate(context_utts[b_id]):
                for j, ele in enumerate(row):
                    new_array[i, j] = ele
            vec_context[b_id, 0:vec_context_lens[b_id], :] = new_array

        return Pack(contexts=vec_context, context_lens=vec_context_lens,
                    outputs=vec_outs, output_lens=vec_out_lens,
                    metas=metas)

# PTB
class PTBDataLoader(DataLoader):

    def __init__(self, name, data, config, max_utt_len=-1):
        super(PTBDataLoader, self).__init__(name, fix_batch=config.fix_batch)
        self.max_utt_size = config.max_utt_len if max_utt_len == -1 else max_utt_len
        self.data = self.pad_data(data)
        self.data_size = len(self.data)
        all_lens = [len(line.utt) for line in self.data]
        print("Max len %d and min len %d and avg len %f" % (np.max(all_lens),
                                                            np.min(all_lens),
                                                            float(np.mean(all_lens))))
        if config.fix_batch:
            self.indexes = list(np.argsort(all_lens))
        else:
            self.indexes = list(range(len(self.data)))

    def pad_data(self, data):
        for l in data:
            l['utt'] = self.pad_to(self.max_utt_size, l.utt, do_pad=False)
        return data

    def _prepare_batch(self, selected_index):
        rows = [self.data[idx] for idx in selected_index] # 通过 selected_index 从 self.data 中获取当前批次的样本数据。
        input_lens = np.array([len(row.utt) for row in rows], dtype=np.int32) # 计算每个样本的实际长度，并将其转换为 numpy 数组。
        max_len = np.max(input_lens) # 获取当前批次中样本的最大长度，用于后续创建输入矩阵。
        inputs = np.zeros((self.batch_size, max_len), dtype=np.int32) # 创建一个大小为 (batch_size, max_len) 的零矩阵，用于存储批次中的所有输入数据。
        for idx, row in enumerate(rows):
            inputs[idx, 0:input_lens[idx]] = row.utt

        return Pack(outputs=inputs, output_lens=input_lens, metas=[data["meta"] for data in rows])
    # 返回一个 Pack 对象，包含以下内容：
    # outputs=inputs：批次的输入数据，大小为 (batch_size, max_len)。
    # output_lens=input_lens：当前批次中每个输入的实际长度，用于后续操作（如 RNN 中忽略填充部分的计算）。
    # metas=[data["meta"] for data in rows]：将当前批次中每个样本的元信息（meta）提取出来，方便后续处理。
