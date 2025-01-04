# @Time    : 9/20/17 2:15 PM
# @Author  : Tiancheng Zhao
import torch
import torch.nn as nn
from dgmvae.enc2dec.base_modules import BaseRNN
import torch.nn.functional as F


class EncoderRNN(BaseRNN):
    """
    Applies a multi-layer RNN to an input sequence.
    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        n_layers (int, optional): number of recurrent layers (default: 1)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        variable_lengths (bool, optional): if use variable length RNN (default: False)
    Inputs: inputs, input_lengths
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is a list of token IDs.
        - **input_lengths** (list of int, optional): list that contains the lengths of sequences
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)
    Outputs: output, hidden
        - **output** (batch, seq_len, hidden_size): tensor containing the encoded features of the input sequence
        - **hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the features in the hidden state `h`
    Examples::
         >>> encoder = EncoderRNN(input_vocab, max_seq_length, hidden_size)
         >>> output, hidden = encoder(input)
    """

    def __init__(self, input_size, hidden_size,
                 input_dropout_p=0, dropout_p=0,
                 n_layers=1, rnn_cell='gru',
                 variable_lengths=False, bidirection=False):
        # input_size：输入特征的维度，通常是词嵌入的大小。
        # hidden_size：隐藏层的特征数量，这个值控制了RNN隐层的容量和学习能力。
        # input_dropout_p：输入序列的 dropout 概率，用于防止过拟合。
        # dropout_p：输出序列的 dropout 概率。
        # n_layers：RNN的层数，默认为1。
        # rnn_cell：RNN单元的类型，默认为 GRU (Gated Recurrent Unit)，可以是 lstm、gru 等。
        # variable_lengths：是否使用变长输入序列（在处理变长序列时有用），默认为 False。
        # bidirection：是否使用双向 RNN，默认为 False。

        super(EncoderRNN, self).__init__(-1, input_size, hidden_size,
                                         input_dropout_p, dropout_p, n_layers,
                                         rnn_cell, bidirection) # -1表示占位符，vocab_size，后续修改BaseRNN后可以删掉它

        self.variable_lengths = variable_lengths
        self.output_size = hidden_size*2 if bidirection else hidden_size

    def forward(self, input_var, input_lengths=None, init_state=None):
        """
        Applies a multi-layer RNN to an input sequence.
        Args:
            input_var (batch, seq_len, embedding size): tensor containing the features of the input sequence.
            input_lengths (list of int, optional): A list that contains the lengths of sequences
              in the mini-batch
        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h
        """
        # input_var：输入张量，形状为 (batch, seq_len, embedding_size)，包含输入序列的特征。
        # input_lengths：输入序列的长度列表，仅在使用变长输入序列时需要。
        # init_state：初始的隐藏状态，如果没有提供则默认用全零初始化。
        embedded = self.input_dropout(input_var) # 对输入应用 dropout，增加模型的泛化能力，防止过拟合。
        # 变长序列处理，后续可以考虑去掉
        if self.variable_lengths: 
            embedded = nn.utils.rnn.pack_padded_sequence(embedded,
                                                         input_lengths,
                                                         batch_first=True)
        # 如果提供了 init_state，则会使用它作为初始隐藏状态；否则使用默认的初始化。
        if init_state is not None:
            output, hidden = self.rnn(embedded, init_state)
        else:
            output, hidden = self.rnn(embedded)
        # 变长序列处理，后续可以考虑去掉
        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output,
                                                         batch_first=True)
        return output, hidden


class RnnUttEncoder(nn.Module):
    def __init__(self, utt_cell_size, dropout,
                 rnn_cell='gru', bidirection=True, use_attn=False,
                 embedding=None, vocab_size=None, embed_dim=None,
                 feat_size=0):
        super(RnnUttEncoder, self).__init__()
        self.bidirection = bidirection
        self.utt_cell_size = utt_cell_size

        if embedding is None:
            self.embed_size = embed_dim
            self.embedding = nn.Embedding(vocab_size, embed_dim)
        else:
            self.embedding = embedding
            self.embed_size = embedding.embedding_dim

        self.rnn = EncoderRNN(self.embed_size+feat_size,
                              utt_cell_size, 0.0, dropout,
                              rnn_cell=rnn_cell, variable_lengths=False,
                              bidirection=bidirection)

        self.multipler = 2 if bidirection else 1
        self.output_size = self.utt_cell_size * self.multipler
        self.use_attn = use_attn
        self.feat_size = feat_size
        if use_attn:
            self.key_w = nn.Linear(self.utt_cell_size*self.multipler,
                                   self.utt_cell_size)
            self.query = nn.Linear(self.utt_cell_size, 1)

    def forward(self, utterances, feats=None, init_state=None, return_all=False):
        batch_size = int(utterances.size()[0])
        max_ctx_lens = int(utterances.size()[1])
        max_utt_len = int(utterances.size()[2])

        # repeat the init state
        if init_state is not None:
            init_state = init_state.repeat(1, max_ctx_lens, 1)

        # get word embeddings
        flat_words = utterances.view(-1, max_utt_len)
        words_embeded = self.embedding(flat_words)

        if feats is not None:
            flat_feats = feats.view(-1, 1)
            flat_feats = flat_feats.unsqueeze(1).repeat(1, max_utt_len, 1)
            words_embeded = torch.cat([words_embeded, flat_feats], dim=2)

        enc_outs, enc_last = self.rnn(words_embeded, init_state=init_state)

        if self.use_attn:  # weighted add enc_outs, whose weights are calculated by attention...
            fc1 = torch.tanh(self.key_w(enc_outs))
            attn = self.query(fc1).squeeze(2)
            attn = F.softmax(attn, attn.dim()-1).unsqueeze(2)
            utt_embedded = attn * enc_outs
            utt_embedded = torch.sum(utt_embedded, dim=1)
        else:
            attn = None
            utt_embedded = enc_last.transpose(0, 1).contiguous()
            utt_embedded = utt_embedded.view(-1, self.output_size)

        utt_embedded = utt_embedded.view(batch_size, max_ctx_lens, self.output_size)

        if return_all:
            return utt_embedded, enc_outs, enc_last, attn
        else:
            return utt_embedded
