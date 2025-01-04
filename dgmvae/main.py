# -*- coding: utf-8 -*-
# author: Tiancheng Zhao
from __future__ import print_function
import numpy as np
from dgmvae.models.model_bases import summary
import torch
from dgmvae.dataset.corpora import PAD, EOS, EOT
from dgmvae.enc2dec.decoders import TEACH_FORCE, GEN, DecoderRNN
from dgmvae.utils import get_dekenize, experiment_name, kl_anneal_function
import os
from collections import defaultdict
import logging
from dgmvae import utt_utils

logger = logging.getLogger()

# 用于管理和记录损失（loss），特别是在训练过程中，它帮助存储、打印、清除和计算损失的平均值。
class LossManager(object):
    def __init__(self):
        self.losses = defaultdict(list) # 初始化一个字典，默认值为列表，用于存储每个损失的列表
        self.backward_losses = [] # 存储用于反向传播的损失

    def add_loss(self, loss): # 该方法用于将一个批次中的损失添加到 self.losses 字典中。loss 是一个字典，通常包含多个不同类型的损失
        for key, val in loss.items():
            if val is not None and type(val) is not bool: # 检查损失值 val 是否有效（非 None 且非布尔类型）
                self.losses[key].append(val.item()) # 将损失值转换为标量（val.item()）并追加到对应键（key）的列表中。

    def add_backward_loss(self, loss): # 一个总体的损失值
        self.backward_losses.append(loss.item()) # 将反向传播损失值添加到 backward_losses 列表中

    def clear(self): # 该方法用于清空损失记录，将 self.losses 重置为空字典，将 self.backward_losses 重置为空列表。
        self.losses = defaultdict(list)
        self.backward_losses = []

    def pprint(self, name, window=None, prefix=None): # 该方法用于打印损失的摘要。
        str_losses = [] # 用于存储损失的字符串列表
        for key, loss in self.losses.items():
            if loss is None:
                continue
            # 如果提供了 window 参数，则计算最近 window 个损失的平均值，否则计算所有损失的平均值。
            avg_loss = np.average(loss) if window is None else np.average(loss[-window:])
            # 将损失名称和平均损失（保留三位小数）格式化后添加到 str_losses 列表中。
            str_losses.append("{} {:.3f}".format(key, avg_loss))
            # 如果损失名称包含 nll 且没有计算 PPL，则根据负对数似然损失计算并添加困惑度（PPL）。
            if 'nll' in key and 'PPL' not in self.losses:
                str_losses.append("PPL {:.3f}".format(np.exp(avg_loss)))
        # 如果提供了 prefix，则将其与损失名称一起返回。
        if prefix:
            return "{}: {} {}".format(prefix, name, " ".join(str_losses))
        # 如果没有提供 prefix，则仅返回损失名称和损失值字符串。
        else:
            return "{} {}".format(name, " ".join(str_losses))
    # 该方法返回一个包含每个损失的平均值的字典。
    def return_dict(self, window=None):
        ret_losses = {} #  一个字典，用于存储每个损失的平均值。
        for key, loss in self.losses.items():
            if loss is None:
                continue
            avg_loss = np.average(loss) if window is None else np.average(loss[-window:])
            ret_losses[key] = avg_loss.item()
            if 'nll' in key and 'PPL' not in self.losses:
                ret_losses[key.split("nll")[0] + 'PPL'] = np.exp(avg_loss).item()
        return ret_losses
    # 该方法返回 self.backward_losses 列表中所有损失的平均值。
    def avg_loss(self):
        return np.mean(self.backward_losses)
    
# 这个函数 adjust_learning_rate 旨在调整优化器（optimizer）的学习率。具体来说，它是按照一定的衰减率（decay_rate）来减少学习率。
def adjust_learning_rate(optimizer, last_lr, decay_rate=0.5):
    # optimizer：优化器对象，通常是 torch.optim 中的某个优化器（例如，SGD、Adam 等）。这个优化器控制模型的参数更新。
    # last_lr：当前的学习率（或上一轮的学习率）。
    # decay_rate：学习率的衰减因子，默认为 0.5，意味着每次调用该函数时，学习率将减少到原来的一半。
    lr = last_lr * decay_rate # 这行代码计算新的学习率 lr，即用当前的学习率（last_lr）乘以衰减因子 decay_rate。
    print('New learning rate=', lr)
    # 是一个包含优化器中所有参数组（parameter groups）的列表。每个参数组是一个字典，包含该组的参数及其超参数（如学习率）。
    for param_group in optimizer.param_groups:
        # 这行代码将每个参数组的学习率更新为原学习率乘以衰减因子，从而实现学习率的衰减。
        param_group['lr'] = param_group['lr'] * decay_rate  # all decay half
    return lr
# 这个 get_sent 函数用于从模型的输出中获取生成的句子（例如，文本生成任务中的预测句子）。它可以处理注意力机制（Attention Mechanism），并返回解码后的文本以及与每个词的注意力分布相关的附加信息。
def get_sent(model, de_tknize, data, b_id, attn=None, attn_ctx=None, stop_eos=True, stop_pad=True):
    # model: 训练好的模型，提供词汇表（model.vocab）和其他用于生成文本的信息。
    # de_tknize: 解码函数（tokenizer），用于将词ID转换为词语（或字符）。
    # data: 生成的预测数据（通常是模型输出的token ID），形状为 (batch_size, seq_length)，表示多个序列的预测。
    # b_id: 批次中的样本索引，表示我们正在处理的具体句子（或样本）。
    # attn: 可选的注意力权重，通常是由注意力机制生成的，用于可视化模型的注意力分布。
    # attn_ctx: 可选的上下文信息（例如，源句子），它与注意力一起用于生成翻译或其他文本。
    # stop_eos: 是否在遇到 EOS（结束符）时停止生成。默认为 True，即遇到结束符就停止。
    # stop_pad: 是否在遇到 PAD（填充符）时停止生成。默认为 True，即遇到填充符就停止。

    # ws 用来保存生成的词（或标记）。生成的每个词都将依次存储在 ws 中。
    ws = []
    # attn_ws 用来保存与注意力相关的信息，即每个词的注意力权重。
    attn_ws = []
    # has_attn 判断是否有注意力信息。如果 attn 和 attn_ctx 都不为 None，表示模型生成时使用了注意力机制。
    has_attn = attn is not None and attn_ctx is not None
    # 遍历数据生成词
    for t_id in range(data.shape[1]):
        w = model.vocab[data[b_id, t_id]]
        # 处理注意力信息
        if has_attn:
            a_val = np.max(attn[b_id, t_id])
            if a_val > 0.1:
                a = np.argmax(attn[b_id, t_id])
                attn_w = model.vocab[attn_ctx[b_id, a]]
                attn_ws.append("{}({})".format(attn_w, a_val))
        # 处理停止条件
        if (stop_eos and w in [EOS, EOT]) or (stop_pad and w == PAD):
            if w == EOT:
                ws.append(w)
            break
        # 处理非填充符的词
        if w != PAD:
            ws.append(w)
    # 处理注意力输出
    att_ws = "Attention: {}".format(" ".join(attn_ws)) if attn_ws else ""
    # 返回解码后的句子和注意力信息
    if has_attn:
        return de_tknize(ws), att_ws
    else:
        try:
            return de_tknize(ws), ""
        except:
            return " ".join(ws), ""

# train 函数用于训练给定的模型（model），它通过多个训练周期（epochs）在提供的训练数据（train_feed）、验证数据（valid_feed）和测试数据（test_feed）上进行训练和评估。函数还包括了早期停止、学习率调整等功能。
def train(model, train_feed, valid_feed, test_feed, config, evaluator, gen=None):
    # model: 要训练的模型。
    # train_feed: 用于训练的输入数据（通常是一个数据生成器或数据加载器）。
    # valid_feed: 用于验证的输入数据。
    # test_feed: 用于测试的输入数据。
    # config: 配置对象，包含训练的相关参数（如学习率、最大训练周期、早停条件等）。
    # evaluator: 用于评估模型输出的评估器。
    # gen: 一个生成函数，用于生成模型的输出并评估其性能。如果不提供，会默认使用 generate 函数。
    if gen is None:
        gen = generate
    # 设置早停策略的耐心值。如果验证损失在 patience 次迭代内没有改善，训练将停止。
    patience = 10  # wait for at least 10 epoch before stop
    # 用于记录验证集上的损失，控制早停。
    valid_loss_threshold = np.inf
    best_valid_loss = np.inf
    valid_loss_record = []
    learning_rate = config.init_lr
    batch_cnt = 0
    # 从模型中获取优化器。
    optimizer = model.get_optimizer(config)
    # 记录已完成的训练周期数。
    done_epoch = 0
    train_loss = LossManager()
    # 将模型设置为训练模式，启用 dropout 等训练特性。
    model.train()
    logger.info(summary(model, show_weights=False))
    logger.info("**** Training Begins ****")
    logger.info("**** Epoch 0/{} ****".format(config.max_epoch))

    while True:
        # 初始化一个新的训练周期，并且可以选择打乱数据。
        train_feed.epoch_init(config, shuffle=True)
        while True:
            # 从训练数据中获取下一个批次，直到没有更多数据为止（batch 为 None 时结束）。
            batch = train_feed.next_batch()
            if batch is None:
                break
            # 在每次训练之前，清空之前计算的梯度。
            optimizer.zero_grad()

            # 如果设置了 flush_valid，表示需要重置一些验证相关的信息。
            if model.flush_valid:
                logger.info("Flush previous valid loss")
                best_valid_loss = np.inf
                model.flush_valid = False
                valid_loss_record = []
                optimizer = model.get_optimizer(config)
                logger.info("Recovering the learning rate to " + str(config.init_lr))
                learning_rate = config.init_lr
                for param_group in optimizer.param_groups:  # recover to the initial learning rate
                    param_group['lr'] = config.init_lr
                # and loading the best model
                logger.info("Load previous best model")

                model_file = os.path.join(config.session_dir, "model")

                if os.path.exists(model_file):
                    if config.model == "GMVAE_pretrain_and_fb":
                        pre_state_dict = torch.load(model_file)
                        state_dict = model.state_dict()
                        for key in state_dict:
                            if "dec" in key:
                                continue
                            state_dict[key].copy_(pre_state_dict[key].data)
                    else:
                        model.load_state_dict(torch.load(model_file))

                # Draw pics:
                # print("Draw pics!")
                # utt_utils.draw_pics(model, test_feed, config, -1, num_batch=5, shuffle=True, add_text=False)  # (num_batch * 50) points
                # model.train()

            # 计算当前批次的损失。TEACH_FORCE 是指强制使用教师信号进行训练。
            loss = model(batch, mode=TEACH_FORCE)
            # 计算梯度
            model.backward(batch_cnt, loss, step=batch_cnt)
            # 防止梯度爆炸，通过剪切梯度来限制梯度的最大范数。
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, norm_type=2)
            # 更新模型的参数。
            optimizer.step()
            batch_cnt += 1
            # 记录每个批次的损失。
            train_loss.add_loss(loss)
            # 每 config.print_step 批次打印一次训练的损失信息。
            if batch_cnt % config.print_step == 0:
                logger.info(train_loss.pprint("Train", window=config.print_step,
                                              prefix="{}/{}-({:.3f})".format(batch_cnt % (config.ckpt_step+1),
                                                                         config.ckpt_step,
                                                                         model.kl_w)))
            # 每 config.ckpt_step 批次，进行一次评估。
            if batch_cnt % config.ckpt_step == 0:
                logger.info("\n=== Evaluating Model ===")
                logger.info(train_loss.pprint("Train"))
                done_epoch += 1

                # validation 在验证集上评估模型的性能。
                valid_loss, valid_resdict = validate(model, valid_feed, config, batch_cnt)
                # 可视化生成的结果（如果配置中要求）。
                if 'draw_points' in config and config.draw_points:
                    utt_utils.draw_pics(model, valid_feed, config, batch_cnt)

                # generating
                # gen：如果定义了生成函数，进行生成并评估。

                #gen_losses = gen(model, test_feed, config, evaluator, num_batch=config.preview_batch_num)

                # adjust learning rate: 如果验证损失没有改进且符合条件，则调整学习率。
                valid_loss_record.append(valid_loss)
                if config.lr_decay and learning_rate > 1e-6 and valid_loss > best_valid_loss and len(
                        valid_loss_record) - valid_loss_record.index(best_valid_loss) >= config.lr_hold:
                    learning_rate = adjust_learning_rate(optimizer, learning_rate, config.lr_decay_rate)
                    logger.info("Adjust learning rete to {}".format(learning_rate))
                    # logger.info("Reloading the best model.")
                    # model.load_state_dict(torch.load(os.path.join(config.session_dir, "model")))
                # 更新最好的验证损失和耐心值。
                # update early stopping stats
                if valid_loss < best_valid_loss:
                    if valid_loss <= valid_loss_threshold * config.improve_threshold:
                        patience = max(patience,
                                       done_epoch * config.patient_increase)
                        valid_loss_threshold = valid_loss
                        logger.info("Update patience to {}".format(patience))
                    # 如果验证损失更好，则保存当前模型。
                    if config.save_model:
                        logger.info("Model Saved.")
                        torch.save(model.state_dict(),
                                   os.path.join(config.session_dir, "model"))

                    best_valid_loss = valid_loss
                # 如果达到最大训练周期、早停条件或学习率过低，则终止训练。
                if done_epoch >= config.max_epoch \
                        or config.early_stop and patience <= done_epoch or learning_rate <= 1e-6:
                    if done_epoch < config.max_epoch:
                        logger.info("!!Early stop due to run out of patience!!")

                    logger.info("Best validation loss %f" % best_valid_loss)

                    return
                # 训练完成一轮后，恢复训练模式并清空损失记录。
                # exit eval model
                model.train()
                train_loss.clear()
                logger.info("\n**** Epoch {}/{} ****".format(done_epoch,
                                                       config.max_epoch))
# 这个函数 validate 主要用于在训练过程中对模型进行验证，并计算验证集上的损失值。它的作用是在验证数据集上评估模型的性能，返回验证损失，并根据需要输出结果到文件。
def validate(model, valid_feed, config, batch_cnt=None, outres2file=None):
    model.eval() # 切换模型为评估模式（禁用 dropout 等）
    valid_feed.epoch_init(config, shuffle=False) # 初始化验证数据集，设置不进行洗牌
    losses = LossManager() # 创建一个 LossManager 对象来管理损失

    while True:
        batch = valid_feed.next_batch() # 获取一个验证批次
        if batch is None: # 如果没有更多的批次，跳出循环
            break
        loss = model(batch, mode=TEACH_FORCE) # 使用教师强制模式计算模型的损失
        losses.add_loss(loss) # 将当前批次的损失添加到 LossManager 中
        losses.add_backward_loss(model.model_sel_loss(loss, batch_cnt)) # 额外的模型选择损失（可能是正则化项等）

    valid_loss = losses.avg_loss() # 计算平均损失
    if outres2file: # 如果提供了输出文件句柄
        outres2file.write(losses.pprint(valid_feed.name)) # 将损失打印到文件
        outres2file.write("\n")
        outres2file.write("Total valid loss {}".format(valid_loss)) # 写入总的验证损失

    logger.info(losses.pprint(valid_feed.name)) # 打印验证损失信息到日志
    logger.info("Total valid loss {}".format(valid_loss)) # 打印总的验证损失

    res_dict = losses.return_dict() # 获取所有损失的字典（包含各个损失项的平均值）

    return valid_loss, res_dict

# 需要根据最后的评估结果修改
# generate 函数的主要目的是生成模型的输出（如序列生成或预测），并将结果与真实标签对比，输出到日志或文件。它通常用于生成序列数据，如自然语言生成、时间序列预测等任务，或者生成分类标签，并通过评价器（evaluator）进行评估。
def generate(model, data_feed, config, evaluator, num_batch=1, dest_f=None):
    model.eval()
    de_tknize = get_dekenize()

    def write(msg):
        if msg is None or msg == '':
            return
        if dest_f is None:
            logger.info(msg)
        else:
            dest_f.write(msg + '\n')

    data_feed.epoch_init(config, shuffle=num_batch is not None)
    evaluator.initialize()
    logger.info("Generation: {} batches".format(data_feed.num_batch
                                                if num_batch is None
                                                else num_batch))
    while True:
        batch = data_feed.next_batch()
        if batch is None or (num_batch is not None
                             and data_feed.ptr > num_batch):
            break
        outputs, labels = model(batch, mode=GEN, gen_type=config.gen_type)

        # move from GPU to CPU
        labels = labels.cpu()
        pred_labels = [t.cpu().data.numpy() for t in
                       outputs[DecoderRNN.KEY_SEQUENCE]]
        pred_labels = np.array(pred_labels, dtype=int).squeeze(-1).swapaxes(0,1)
        true_labels = labels.data.numpy()
        # get attention if possible
        if config.use_attn or config.use_ptr:
            pred_attns = [t.cpu().data.numpy() for t in outputs[DecoderRNN.KEY_ATTN_SCORE]]
            pred_attns = np.array(pred_attns, dtype=float).squeeze(2).swapaxes(0,1)
        else:
            pred_attns = None

        # get last 1 context
        ctx = batch.get('contexts')
        ctx_len = batch.get('context_lens')
        domains = batch.domains

        # logger.info the batch in String.
        for b_id in range(pred_labels.shape[0]):
            pred_str, attn = get_sent(model, de_tknize, pred_labels, b_id, attn=pred_attns)
            true_str, _ = get_sent(model, de_tknize, true_labels, b_id)
            prev_ctx = ""
            if ctx is not None:
                ctx_str, _ = get_sent(model, de_tknize, ctx[:, ctx_len[b_id]-1, :], b_id)
                prev_ctx = "Source: {}".format(ctx_str)

            domain = domains[b_id]
            evaluator.add_example(true_str, pred_str, domain)
            if num_batch is None or num_batch <= 2:
                write(prev_ctx)
                write("{}:: True: {} ||| Pred: {}".format(domain, true_str, pred_str))
                if attn:
                    write("[[{}]]".format(attn))

    write(evaluator.get_report(include_error=dest_f is not None))
    logger.info("Generation Done")



