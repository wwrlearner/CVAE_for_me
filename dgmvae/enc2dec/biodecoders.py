import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

class DecoderRNN(nn.Module):
    def __init__(self, output_size, input_size, hidden_size, n_layers=1, rnn_cell='gru',
                 input_dropout_p=0, dropout_p=0, use_attention=False, attn_mode='cat',
                 attn_size=None, use_gpu=True, init_range=0.1):
        super(DecoderRNN, self).__init__()

        self.output_size = output_size  # 输出的大小，一般为神经元数目
        self.use_attention = use_attention  # 是否使用注意力机制
        self.hidden_size = hidden_size  # RNN隐层的大小
        self.input_size = input_size  # 输入特征的大小
        self.use_gpu = use_gpu  # 是否使用GPU

        # 输入特征的dropout
        self.input_dropout = nn.Dropout(p=input_dropout_p)

        # 定义RNN类型，可以选择LSTM或GRU
        if rnn_cell.lower() == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p)
        elif rnn_cell.lower() == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p)
        else:
            raise ValueError("Unsupported RNN Cell: {}".format(rnn_cell))

        # 定义全连接层，将隐状态投影到输出空间
        self.project = nn.Linear(hidden_size, output_size)

        # 初始化网络参数
        for para in self.parameters():
            nn.init.uniform_(para.data, -init_range, init_range)

    def forward_step(self, max_len, input_var, hidden, encoder_outputs=None, latent_variable=None):
        # input_var: 当前时间步的输入，形状为(batch_size, 1, input_size)
        # hidden: 上一时间步的隐状态
        # latent_variable: 潜变量，如果存在则拼接到输入中
        
        embedded = self.input_dropout(input_var)  # 对输入进行dropout处理
        # 这个地方后续可以作为一个接口探究
        #if latent_variable is not None:
            # 将潜变量与输入拼接
            #latent_variable = latent_variable.unsqueeze(1).repeat(1, embedded.size(1), 1)
            #embedded = torch.cat([embedded, latent_variable], dim=-1)

        # 通过RNN得到输出和新的隐状态
        #print(embedded.size())
        #print(hidden.size())
        output, hidden = self.rnn(embedded, hidden)
        # 如果使用注意力机制，可以在这里添加注意力层（对于神经元信号生成，通常不需要）
        if self.use_attention and encoder_outputs is not None:
            # 注意力机制可以在这里加入，若适用
            pass
        #print(output.size())
        # 投影到输出空间
        logits = self.project(output.contiguous().view(-1, 1, self.hidden_size)) # 修改输出的维度
        #+print(logits.size())
        return logits, output, hidden

    def forward(self, max_len, batch_size, inputs=None, init_state=None, latent_variable=None):
        # 初始化隐状态
        if init_state is None:
            decoder_hidden = None
        else:
            if isinstance(self.rnn, nn.LSTM):
                decoder_hidden = (init_state[0].repeat(self.rnn.num_layers, 1, 1),
                                  init_state[1].repeat(self.rnn.num_layers, 1, 1))
            else:
                decoder_hidden = init_state.repeat(self.rnn.num_layers, 1, 1)

        decoder_outputs = []  # 存储每个时间步的输出

        # 手动展开RNN，以支持自由运行模式
        for di in range(max_len):
            if inputs is not None:
                decoder_input = inputs[:, di, :].unsqueeze(1)  # 获取当前时间步的输入
            else:
                # 如果没有提供输入（自由运行模式），使用上一个时间步的输出作为输入
                if di == 0:
                    # 在第一时间步，输入全为0
                    decoder_input = torch.zeros(batch_size, 1, self.input_size)
                    if self.use_gpu:
                        decoder_input = decoder_input.cuda()
                else:
                    # 使用上一时间步的输出作为当前输入
                    decoder_input = output

            # 前向计算当前时间步的输出
            logits, output, decoder_hidden = self.forward_step(max_len, decoder_input, decoder_hidden, latent_variable=latent_variable)
            # 存储当前时间步的输出
            decoder_outputs.append(logits)

        # 将所有时间步的输出拼接

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        return decoder_outputs, decoder_hidden

