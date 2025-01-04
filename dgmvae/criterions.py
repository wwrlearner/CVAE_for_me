# -*- coding: utf-8 -*-
from __future__ import print_function
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import numpy as np
import torch
from dgmvae.utils import INT, FLOAT, LONG, cast_type
import logging


class L2Loss(_Loss):

    logger = logging.getLogger()
    def forward(self, state_a, state_b):
        if type(state_a) is tuple:
            losses = 0.0
            for s_a, s_b in zip(state_a, state_b):
                losses += torch.pow(s_a-s_b, 2)
        else:
            losses = torch.pow(state_a-state_b, 2)
        return torch.mean(losses)

class NLLEntropy(_Loss):

    logger = logging.getLogger()
    def __init__(self, padding_idx, config, rev_vocab=None, key_vocab=None, avg_type="seq"):
        super(NLLEntropy, self).__init__()
        self.padding_idx = padding_idx
        # self.avg_type = config.avg_type
        self.avg_type = avg_type

        if rev_vocab is None or key_vocab is None:
            self.weight = None
        else:
            self.logger.info("Use extra cost for key words")
            weight = np.ones(len(rev_vocab))
            for key in key_vocab:
                weight[rev_vocab[key]] = 10.0
            self.weight = cast_type(torch.from_numpy(weight), FLOAT,
                                    config.use_gpu)

        if self.avg_type == "word":
            self.nll_loss = nn.NLLLoss(weight=self.weight, ignore_index=self.padding_idx)
        elif self.avg_type == 'real_word':
            self.nll_loss = nn.NLLLoss(weight=self.weight, ignore_index=self.padding_idx, reduction=None)
        elif self.avg_type == "seq":
            self.nll_loss = nn.NLLLoss(weight=self.weight, ignore_index=self.padding_idx, reduction='sum')
        elif self.avg_type is None:
            self.nll_loss = nn.NLLLoss(weight=self.weight, ignore_index=self.padding_idx, reduction='sum')
        else:
            raise NotImplementedError("Unknown avg type")


    def forward(self, net_output, labels):
        batch_size = net_output.size(0)
        input = net_output.view(-1, net_output.size(-1))
        target = labels.view(-1)

        if self.avg_type is None:
            loss = self.nll_loss(input, target)
        elif self.avg_type == 'seq':
            loss = self.nll_loss(input, target)
            loss = loss / batch_size
        elif self.avg_type == 'real_word':
            loss = self.nll_loss(input, target)
            loss = loss.view(-1, net_output.size(1))
            loss = torch.sum(loss, dim=1)
            word_cnt = torch.sum(torch.sign(labels), dim=1).float()
            loss = loss/word_cnt
            loss = torch.mean(loss)
        elif self.avg_type == 'word':
            loss = self.nll_loss(input, target)
        else:
            raise ValueError("Unknown avg type")

        return loss
class TimeSeriesNLLLoss(nn.Module):
    def __init__(self, avg_type="seq"):
        super(TimeSeriesNLLLoss, self).__init__()
        
        # avg_type：选择不同的平均策略 ("seq", "time", None等)
        self.avg_type = avg_type
        
        # 使用简单的 NLLLoss
        self.nll_loss = nn.NLLLoss(reduction='sum')

    def forward(self, net_output, labels):
        """
        :param net_output: 模型输出的概率分布，形状应为 (batch_size, seq_len, num_classes)
        :param labels: 真实标签，形状应为 (batch_size, seq_len)
        :return: 计算得到的损失
        """
        batch_size = net_output.size(0)
        
        # 将输出和标签展平，便于计算
        input = net_output.view(-1, net_output.size(-1))  # (batch_size * seq_len, num_classes)
        target = labels.view(-1)  # (batch_size * seq_len)
        
        # 计算 NLLLoss
        loss = self.nll_loss(input, target)
        
        # 根据 avg_type 选择如何归一化损失
        if self.avg_type == 'seq':
            loss = loss / batch_size
        elif self.avg_type == 'time':
            # "time" 表示每个时间步的平均损失
            loss = loss / torch.sum(torch.sign(target))  # 统计非零标签的数量（假设标签是非负的）
        elif self.avg_type is None:
            pass  # 默认不进行平均，直接返回总损失
        else:
            raise ValueError("Unknown avg type")
        
        return loss

class TimeSeriesLossMSE(nn.Module):
    def __init__(self, avg_type="seq"):
        super(TimeSeriesLossMSE, self).__init__()

        # avg_type：选择不同的平均策略 ("seq", "time", None等)
        self.avg_type = avg_type
        
        # 使用 MSELoss 作为回归损失函数
        self.mse_loss = nn.MSELoss(reduction='sum')

    def forward(self, net_output, labels):
        """
        :param net_output: 模型输出，形状应为 (batch_size, seq_len, num_features)
        :param labels: 真实标签，形状应为 (batch_size, seq_len, num_features)
        :return: 计算得到的损失
        """
        batch_size = net_output.size(0)
        
        # 将输出和标签展平，便于计算
        input = net_output.view(-1, net_output.size(-1))  # (batch_size * seq_len, num_features)
        target = labels.view(-1, labels.size(-1))  # (batch_size * seq_len, num_features)
        
        # 计算 MSELoss
        loss = self.mse_loss(input, target)
        
        # 根据 avg_type 选择如何归一化损失
        if self.avg_type == 'seq':
            loss = loss / batch_size
        elif self.avg_type == 'time':
            # "time" 表示每个时间步的平均损失
            loss = loss / torch.sum(torch.sign(target))  # 统计非零标签的数量
        elif self.avg_type is None:
            pass  # 默认不进行平均，直接返回总损失
        else:
            raise ValueError("Unknown avg type")
        
        return loss
    

class PoissonNLLLoss(nn.Module):
    def __init__(self, avg_type="seq"):
        super(PoissonNLLLoss, self).__init__()
        self.avg_type = avg_type

    def forward(self, net_output, labels):
        """
        计算生成的 Poisson 信号与真实信号之间的负对数似然损失。
        
        net_output: 模型生成的 poisson 信号均值 (batch_size, t, neuron)
        labels: 真实的 poisson spike 信号 (batch_size, t, neuron)
        """
        # 计算泊松分布的 NLL 损失
        # log_input=False, 因为 net_output 是泊松分布的均值 lambda
        batch_size, t, neuron = net_output.size()
        net_output = net_output.view(-1, neuron)  # (batch_size * t, neuron)
        labels = labels.view(-1, neuron)  # (batch_size * t, neuron)
        loss = F.poisson_nll_loss(net_output, labels, log_input=False, full=False, eps=1e-8, reduction='sum')

        # 根据 avg_type 来决定如何对损失进行平均
        batch_size = net_output.size(0)
        if self.avg_type == 'seq':
            loss = loss / batch_size
        else:
            raise ValueError(f"Unknown avg_type: {self.avg_type}")

        return loss



class BowEntropy(_Loss):
    logger = logging.getLogger()

    def __init__(self, padding_idx, config, rev_vocab=None, key_vocab=None):
        super(BowEntropy, self).__init__()
        self.padding_idx = padding_idx
        # self.avg_type = config.avg_type

        if rev_vocab is None or key_vocab is None:
            self.weight = None
        else:
            self.logger.info("Use extra cost for key words")
            weight = np.ones(len(rev_vocab))
            for key in key_vocab:
                weight[rev_vocab[key]] = 10.0
            self.weight = cast_type(torch.from_numpy(weight), FLOAT,
                                    config.use_gpu)

    def forward(self, net_output, labels):
        batch_size = net_output.size(0)
        # input = net_output.view(-1, net_output.size(-1))
        # print(input.size())
        # print(net_output.size())
        # print(labels.size())

        input = net_output.unsqueeze(1).repeat(1, labels.size(-1), 1).view(-1, net_output.size(-1))
        target = labels.view(-1)

        # print(input.size())
        # print(target.size())

        loss = F.nll_loss(input, target, size_average=False,
                          ignore_index=self.padding_idx,
                          weight=self.weight)
        loss = loss / batch_size

        return loss

class Perplexity(_Loss):
    logger = logging.getLogger()
    def __init__(self, padding_idx, config, rev_vocab=None, key_vocab=None):
        super(Perplexity, self).__init__()
        self.padding_idx = padding_idx

        if rev_vocab is None or key_vocab is None:
            self.weight = None
        else:
            self.logger.info("Use extra cost for key words")
            weight = np.ones(len(rev_vocab))
            for key in key_vocab:
                weight[rev_vocab[key]] = 10.0
            self.weight = cast_type(torch.from_numpy(weight), FLOAT,
                                    config.use_gpu)

    def forward(self, net_output, labels):
        # batch_size = net_output.size(0)
        input = net_output.view(-1, net_output.size(-1))
        target = labels.view(-1)
        loss = F.nll_loss(input, target, ignore_index=self.padding_idx, weight=self.weight)
        return torch.exp(loss)

class NormKLLoss(_Loss):
    def __init__(self, unit_average=False):
        super(NormKLLoss, self).__init__()
        self.unit_average = unit_average

    def forward(self, recog_mu, recog_logvar, prior_mu, prior_logvar):
        # find the KL divergence between two Gaussian distribution
        loss = 1.0 + (recog_logvar - prior_logvar)
        loss -= torch.div(torch.pow(prior_mu - recog_mu, 2), torch.exp(prior_logvar))
        loss -= torch.div(torch.exp(recog_logvar), torch.exp(prior_logvar))
        if self.unit_average:
            kl_loss = -0.5 * torch.mean(loss, dim=1)
        else:
            kl_loss = -0.5 * torch.sum(loss, dim=1)
        avg_kl_loss = torch.mean(kl_loss)
        return avg_kl_loss

class CatKLLoss(_Loss):
    def __init__(self):
        super(CatKLLoss, self).__init__()

    def forward(self, log_qy, log_py, batch_size=None, unit_average=False, average = True):
        """
        qy * log(q(y)/p(y))
        """
        if log_qy.dim() > 2:
            log_qy = log_qy.squeeze()

        qy = torch.exp(log_qy)
        y_kl = torch.sum(qy * (log_qy - log_py), dim=-1)
        if not average:
            return y_kl
        if unit_average:
            return torch.mean(y_kl)
        else:
            return torch.sum(y_kl)/batch_size

class CrossEntropyoss(_Loss):
    def __init__(self):
        super(CrossEntropyoss, self).__init__()

    def forward(self, log_qy, log_py, batch_size=None, unit_average=False):
        """
        -qy log(qy) + qy * log(q(y)/p(y))
        """
        if log_qy.dim() > 2:
            log_qy = log_qy.squeeze()
        qy = torch.exp(log_qy)
        h_q = torch.sum(-1 * log_qy * qy, dim=1)
        kl_qp = torch.sum(qy * (log_qy - log_py), dim=1)
        cross_ent = h_q + kl_qp
        if unit_average:
            return torch.mean(cross_ent)
        else:
            return torch.sum(cross_ent)/batch_size

class Entropy(_Loss):
    def __init__(self):
        super(Entropy, self).__init__()

    def forward(self, log_qy, batch_size=None, unit_average=False):
        """
        -qy log(qy)
        """
        if log_qy.dim() > 2:
            log_qy = log_qy.squeeze()
        qy = torch.exp(log_qy)
        h_q = torch.sum(-1 * log_qy * qy, dim=1)
        if unit_average:
            return torch.mean(h_q)
        else:
            return torch.sum(h_q) / batch_size




