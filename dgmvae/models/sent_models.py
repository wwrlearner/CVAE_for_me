import torch
import torch.nn as nn
import torch.nn.functional as F
from dgmvae.dataset.corpora import PAD, BOS, EOS, UNK
from torch.autograd import Variable
from dgmvae import criterions
from dgmvae.enc2dec.biodecoders import DecoderRNN
from dgmvae.enc2dec.encoders import EncoderRNN
from dgmvae.utils import INT, FLOAT, LONG, cast_type
from dgmvae import nn_lib
import numpy as np
from dgmvae.models.model_bases import BaseModel
from dgmvae.enc2dec.decoders import GEN, TEACH_FORCE
from dgmvae.utils import Pack, kl_anneal_function, interpolate, idx2onehot
import itertools
import math

class SVAE(BaseModel):
    def __init__(self, corpus, config):
        super(SVAE, self).__init__(config)
        self.vocab = corpus.vocab
        self.rev_vocab = corpus.rev_vocab
        self.vocab_size = len(self.vocab)
        self.embed_size = config.embed_size
        self.max_utt_len = config.max_utt_len
        self.go_id = self.rev_vocab[BOS]
        self.eos_id = self.rev_vocab[EOS]
        self.unk_id = self.rev_vocab[UNK]
        self.num_layer_enc = config.num_layer_enc
        self.num_layer_dec = config.num_layer_dec
        self.dropout = config.dropout
        self.enc_cell_size = config.enc_cell_size
        self.dec_cell_size = config.dec_cell_size
        self.rnn_cell = config.rnn_cell
        self.max_dec_len = config.max_dec_len
        self.use_attn = config.use_attn
        self.beam_size = config.beam_size
        self.utt_type = config.utt_type
        self.bi_enc_cell = config.bi_enc_cell
        self.attn_type = config.attn_type
        self.enc_out_size = self.enc_cell_size * 2 if self.bi_enc_cell else self.enc_cell_size
        self.concat_decoder_input = config.concat_decoder_input if "concat_decoder_input" in config else False
        self.posterior_sample_n = config.post_sample_num if "post_sample_num" in config else 1

        # build model here
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size,
                                      padding_idx=self.rev_vocab[PAD])

        self.x_encoder = EncoderRNN(self.embed_size, self.enc_cell_size,
                                    dropout_p=self.dropout,
                                    rnn_cell=self.rnn_cell,
                                    variable_lengths=self.config.fix_batch,
                                    bidirection=self.bi_enc_cell,
                                    n_layers=self.num_layer_enc)

        self.q_y_mean = nn.Linear(self.enc_out_size, config.latent_size)
        self.q_y_logvar = nn.Linear(self.enc_out_size, config.latent_size)
        self.q_c = nn.Linear(self.enc_out_size, config.k * config.mult_k)

        self.cat_connector = nn_lib.GumbelConnector()
        self.dec_init_connector = nn_lib.LinearConnector(
            config.latent_size + config.k * config.mult_k,
            self.dec_cell_size,
            self.rnn_cell == 'lstm',
            has_bias=False)

        self.decoder = DecoderRNN(self.vocab_size, self.max_dec_len,
                                  self.embed_size + self.config.latent_size if self.concat_decoder_input else self.embed_size,
                                  self.dec_cell_size,
                                  self.go_id, self.eos_id, self.unk_id,
                                  n_layers=self.num_layer_dec, rnn_cell=self.rnn_cell,
                                  input_dropout_p=self.dropout,
                                  dropout_p=self.dropout,
                                  use_attention=self.use_attn,
                                  attn_size=self.enc_cell_size,
                                  attn_mode=self.attn_type,
                                  use_gpu=self.use_gpu,
                                  tie_output_embed=config.tie_output_embed if "tie_output_embed" in config else False,
                                  embedding=self.embedding)

        self.nll_loss = criterions.NLLEntropy(self.rev_vocab[PAD], self.config)
        self.ppl = criterions.Perplexity(self.rev_vocab[PAD], self.config)
        self.cat_kl_loss = criterions.CatKLLoss()
        self.cross_ent_loss = criterions.CrossEntropyoss()
        self.entropy_loss = criterions.Entropy()

        self.log_py = nn.Parameter(torch.log(torch.ones(self.config.latent_size,
                                                        self.config.k) / config.k),
                                   requires_grad=True)

        self.register_parameter('log_py', self.log_py)

        self.log_uniform_y = Variable(torch.log(torch.ones(1) / config.k))
        if self.use_gpu:
            self.log_uniform_y = self.log_uniform_y.cuda()

        self.kl_w = 0.0

        self.return_latent_key = ('log_qy', 'dec_init_state', 'y_ids')

    @staticmethod
    def add_args(parser):
        from dgmvae.utils import str2bool
        # Latent variable:
        parser.add_argument('--latent_size', type=int, default=40, help="The latent size of continuous latent variable.")
        parser.add_argument('--mult_k', type=int, default=20, help="The number of discrete latent variables.")
        parser.add_argument('--k', type=int, default=5, help="The dimension of discrete latent variable.")

        # Network setting:
        parser.add_argument('--rnn_cell', type=str, default='gru')
        parser.add_argument('--embed_size', type=int, default=512)
        parser.add_argument('--utt_type', type=str, default='rnn')
        parser.add_argument('--enc_cell_size', type=int, default=512)
        parser.add_argument('--dec_cell_size', type=int, default=512)
        parser.add_argument('--bi_enc_cell', type=str2bool, default=True)
        parser.add_argument('--num_layer_enc', type=int, default=1)
        parser.add_argument('--num_layer_dec', type=int, default=1)
        parser.add_argument('--use_attn', type=str2bool, default=False)
        parser.add_argument('--attn_type', type=str, default='cat')
        parser.add_argument('--tie_output_embed', type=str2bool, default=True)
        parser.add_argument('--max_utt_len', type=int, default=40)
        parser.add_argument('--max_dec_len', type=int, default=40)
        parser.add_argument('--max_vocab_cnt', type=int, default=10000)

        # Other settings
        parser.add_argument('--use_mutual', type=str2bool, default=False)
        parser.add_argument('--concat_decoder_input', type=str2bool, default=True)
        parser.add_argument('--gmm', type=str2bool, default=False)

        return parser

    def reparameterization(self, mu, logvar, sample=True):
        if self.training or sample:
            std = torch.exp(0.5 * logvar)
            z = self.torch2var(torch.randn(mu.size()))
            z = z * std + mu
            return z
        else:
            return mu

    def model_sel_loss(self, loss, batch_cnt):
        return loss.elbo

    def valid_loss(self, loss, batch_cnt=None, step=None):
        if batch_cnt is not None:
            step = batch_cnt
        if step is not None and 'anneal_function' in self.config:
            vae_kl_weight = kl_anneal_function(self.config.anneal_function, step,
                                               self.config.anneal_k, self.config.anneal_x0)
        else:
            vae_kl_weight = 1.0

        if not self.config.anneal:
            vae_kl_weight = 1.0

        mi_weight = 0.0 if self.config.use_mutual else 1.0
        total_loss = loss.nll + vae_kl_weight * (
                loss.agg_ckl + mi_weight * loss.mi + loss.zkl)

        return total_loss

    def zkl_loss(self, qy_mean, qy_logvar):
        KL_loss = -0.5 * torch.mean(torch.sum((1 + qy_logvar - qy_mean.pow(2) - qy_logvar.exp()), dim=1))
        return KL_loss

    def forward(self, data_feed, mode, gen_type='greedy', sample_n=1, return_latent=False):
        posterior_sample_n = self.posterior_sample_n if self.training else 1
        if isinstance(data_feed, tuple):
            data_feed = data_feed[0]

        batch_size = len(data_feed['output_lens'])
        out_utts = self.np2var(data_feed['outputs'], LONG)

        # output encoder
        output_embedding = self.embedding(out_utts)
        x_outs, x_last = self.x_encoder(output_embedding)
        if type(x_last) is tuple:
            x_last = x_last[0].view(self.num_layer_enc, 1 + int(self.bi_enc_cell), -1, self.enc_cell_size)[-1]
            x_last = x_last.transpose(0, 1).contiguous().view(-1, self.enc_out_size)
        else:
            x_last = x_last.view(self.num_layer_enc, 1 + int(self.bi_enc_cell), -1, self.enc_cell_size)[-1]
            x_last = x_last.transpose(0, 1).contiguous().view(-1,
                                                              self.enc_out_size)

        # posterior network
        qy_mean = self.q_y_mean(x_last)  # batch x (latent_size*mult_k)
        qy_logvar = self.q_y_logvar(x_last)
        q_z = self.reparameterization(qy_mean.repeat(posterior_sample_n, 1),
                                      qy_logvar.repeat(posterior_sample_n, 1),
                                      sample=gen_type != "greedy" or mode != GEN)  # batch x latent_size
        qc_logits = self.q_c(x_last).view(-1, self.config.k)  # batch*mult_k x k
        log_qc = F.log_softmax(qc_logits, qc_logits.dim() - 1)

        # switch that controls the sampling
        sample_y, y_ids = self.cat_connector(qc_logits.repeat(posterior_sample_n, 1),
                                             1.0, self.use_gpu,
                                             hard=not self.training, return_max_id=True)
        # sample_y: [batch*mult_k, k], y_ids: [batch*mult_k, 1]
        sample_y = sample_y.view(-1, self.config.mult_k * self.config.k)
        y_ids = y_ids.view(-1, self.config.mult_k)

        # map sample to initial state of decoder
        dec_init_state = self.dec_init_connector(torch.cat((sample_y, q_z), dim=1))

        # get decoder inputs
        labels = out_utts[:, 1:].contiguous()
        dec_inputs = out_utts[:, 0:-1]

        # decode
        dec_outs, dec_last, dec_ctx = self.decoder(batch_size * posterior_sample_n,
                                                   dec_inputs.repeat(posterior_sample_n, 1),
                                                   dec_init_state,
                                                   mode=mode, gen_type=gen_type,
                                                   beam_size=self.beam_size,
                                                   latent_variable=q_z)

        dec_ctx[DecoderRNN.KEY_LATENT] = y_ids
        # compute loss or return results
        if mode == GEN:
            return dec_ctx, labels
        else:
            # RNN reconstruction
            nll = self.nll_loss(dec_outs, labels.repeat(posterior_sample_n, 1))
            ppl = self.ppl(dec_outs, labels.repeat(posterior_sample_n, 1))
            # Regularization terms
            avg_log_qc = torch.exp(log_qc.view(-1, self.config.mult_k, self.config.k))
            avg_log_qc = torch.log(torch.mean(avg_log_qc, dim=0) + 1e-15)
            agg_ckl = self.cat_kl_loss(avg_log_qc, self.log_uniform_y, batch_size, unit_average=True, average=False)
            agg_ckl = torch.sum(agg_ckl)
            ckl_real = self.cat_kl_loss(log_qc, self.log_uniform_y, batch_size, unit_average=True, average=False)
            ckl_real = torch.sum(torch.mean(ckl_real.view(-1, self.config.mult_k), dim=0))
            zkl = self.zkl_loss(qy_mean, qy_logvar)  # [batch_size x mult_k]
            mi = - torch.sum(torch.exp(avg_log_qc) * avg_log_qc) + torch.sum(torch.exp(log_qc) * log_qc) / batch_size

            results = Pack(nll=nll, agg_ckl=agg_ckl, mi=mi, real_ckl=ckl_real, elbo=nll+ckl_real+zkl, zkl=zkl, PPL=ppl)
            if return_latent:
                results['log_qy'] = log_qc
                results['dec_init_state'] = dec_init_state
                results['y_ids'] = y_ids
            return results

    def sampling(self, batch_size):
        sample_y = torch.randint(0, self.config.k, [batch_size, self.config.mult_k], dtype=torch.long).cuda()
        zs = self.torch2var(torch.randn(batch_size, self.config.latent_size))
        cs = self.torch2var(idx2onehot(sample_y.view(-1), self.config.k)).view(-1, self.config.mult_k * self.config.k)
        dec_init_state = self.dec_init_connector(torch.cat((cs, zs), dim=1))
        _, _, outputs = self.decoder(cs.size(0),
                                      None, dec_init_state,
                                      mode=GEN, gen_type=self.config.gen_type,
                                      beam_size=self.config.beam_size,
                                      latent_variable=zs
                                      )
        return outputs

    def sampling_for_likelihood(self, batch_size, data_feed, sample_num, sample_type="LL"):
        # Importance sampling...
        assert sample_type in ("LL", "logLL")

        # just for calculating log-likelihood
        if type(data_feed) is tuple:
            data_feed = data_feed[0]
        batch_size = len(data_feed['output_lens'])
        out_utts = self.np2var(data_feed['outputs'], LONG)  # batch_size * seq_len
        out_utts = out_utts.repeat(sample_num, 1)

        labels = out_utts[:, 1:].contiguous()
        dec_inputs = out_utts[:, 0:-1]

        output_embedding = self.embedding(out_utts)
        x_outs, x_last = self.x_encoder(output_embedding)
        if type(x_last) is tuple:
            x_last = x_last[0].view(self.num_layer_enc, 1 + int(self.bi_enc_cell), -1, self.enc_cell_size)[-1]
            x_last = x_last.transpose(0, 1).contiguous().view(-1, self.enc_out_size)
        else:
            x_last = x_last.view(self.num_layer_enc, 1 + int(self.bi_enc_cell), -1, self.enc_cell_size)[-1]
            x_last = x_last.transpose(0, 1).contiguous().view(-1,
                                                              self.enc_out_size)

        qy_mean = self.q_y_mean(x_last)  # [batch_size * sample_num, latent_size]
        qy_logvar = self.q_y_logvar(x_last)
        q_z = self.reparameterization(qy_mean, qy_logvar, sample=True)
        # [batch_size * sample_num, latent_size]
        log_qzx = torch.sum(
            - (q_z - qy_mean) * (q_z - qy_mean) / (2 * torch.exp(qy_logvar)) - 0.5 * qy_logvar - 0.5 * math.log(
                math.pi * 2),
            dim=-1)
        log_pz = torch.sum(
            - (q_z) * (q_z) / 2 - 0.5 * math.log(math.pi * 2),
            dim=-1)

        qc_logits = self.q_c(x_last).view(-1, self.config.k)  # batch*mult_k x k
        log_qcx = F.log_softmax(qc_logits, qc_logits.dim() - 1)

        sample_c = torch.multinomial(torch.exp(log_qcx), 1) # .view(-1, self.config.mult_k)  # [batch_size, mult_k]
        log_qcx = torch.sum(torch.gather(log_qcx, 1, sample_c).view(-1, self.config.mult_k), dim=-1)
        sample_c = self.torch2var(idx2onehot(sample_c.view(-1), self.config.k)).view(-1, self.config.mult_k * self.config.k)

        log_pc = math.log(1.0 / self.config.k) * self.config.mult_k


        # Calculate p(x|z)
        dec_init_state = self.dec_init_connector(torch.cat((sample_c, q_z), dim=1))

        dec_outs, dec_last, outputs = self.decoder(sample_c.size(0),
                                                   dec_inputs,
                                                   dec_init_state,
                                                   mode=TEACH_FORCE,
                                                   gen_type=self.config.gen_type,
                                                   beam_size=self.beam_size,
                                                   latent_variable=q_z)

        nll = F.nll_loss(dec_outs.view(-1, dec_outs.size(-1)), labels.view(-1), reduction="none").view(out_utts.size(0),
                                                                                                       -1)
        nll = torch.sum(nll, dim=-1)
        ll = torch.exp(-nll.double() + log_pz.double() + log_pc - log_qzx.double() - log_qcx.double())
        if sample_type == "logLL":
            return (-nll.double() + log_pz.double() + log_pc - log_qzx.double() - log_qcx.double()).view(-1, sample_num)
        else:
            ll = ll.view(-1, sample_num)
        return ll

class DiVAE(BaseModel):
    def __init__(self, corpus, config):
        super(DiVAE, self).__init__(config)
        self.vocab = corpus.vocab
        self.rev_vocab = corpus.rev_vocab
        self.vocab_size = len(self.vocab)
        self.embed_size = config.embed_size
        self.max_utt_len = config.max_utt_len
        self.go_id = self.rev_vocab[BOS]
        self.eos_id = self.rev_vocab[EOS]
        self.unk_id = self.rev_vocab[UNK]
        self.pad_id = self.rev_vocab[PAD]
        self.num_layer_enc = config.num_layer_enc
        self.num_layer_dec = config.num_layer_dec
        self.dropout = config.dropout
        self.enc_cell_size = config.enc_cell_size
        self.dec_cell_size = config.dec_cell_size
        self.rnn_cell = config.rnn_cell
        self.max_dec_len = config.max_dec_len
        self.use_attn = config.use_attn
        self.beam_size = config.beam_size
        self.utt_type = config.utt_type
        self.bi_enc_cell = config.bi_enc_cell
        self.attn_type = config.attn_type
        self.enc_out_size = self.enc_cell_size * 2 if self.bi_enc_cell else self.enc_cell_size
        self.posterior_sample_n = config.post_sample_num if "post_sample_num" in config else 1
        self.concat_decoder_input = config.concat_decoder_input if "concat_decoder_input" in config else False
        self.use_kl = getattr(config, "use_kl", True)

        # build model here
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size,
                                      padding_idx=self.rev_vocab[PAD])

        self.x_encoder = EncoderRNN(self.embed_size, self.enc_cell_size,
                                    dropout_p=self.dropout,
                                    rnn_cell=self.rnn_cell,
                                    variable_lengths=self.config.fix_batch,
                                    bidirection=self.bi_enc_cell,
                                    n_layers=self.num_layer_enc)

        self.q_y = nn.Linear(self.enc_out_size, config.mult_k * config.k)
        self.cat_connector = nn_lib.GumbelConnector()
        self.dec_init_connector = nn_lib.LinearConnector(config.mult_k * config.k,
                                                         self.dec_cell_size,
                                                         self.rnn_cell == 'lstm',
                                                         has_bias=False)

        self.decoder = DecoderRNN(self.vocab_size,
                                  self.max_dec_len,
                                  self.embed_size + self.config.mult_k * self.config.k if self.concat_decoder_input else self.embed_size,
                                  self.dec_cell_size,
                                  self.go_id, self.eos_id, self.unk_id,
                                  n_layers=self.num_layer_dec, rnn_cell=self.rnn_cell,
                                  input_dropout_p=self.dropout,
                                  dropout_p=self.dropout,
                                  use_attention=self.use_attn,
                                  attn_size=self.enc_cell_size,
                                  attn_mode=self.attn_type,
                                  use_gpu=self.use_gpu,
                                  tie_output_embed=config.tie_output_embed,
                                  embedding=self.embedding)

        self.nll_loss = criterions.NLLEntropy(self.rev_vocab[PAD], self.config)
        self.ppl = criterions.Perplexity(self.rev_vocab[PAD], self.config)
        self.cat_kl_loss = criterions.CatKLLoss()
        self.cross_ent_loss = criterions.CrossEntropyoss()
        self.entropy_loss = criterions.Entropy()
        self.log_py = nn.Parameter(torch.log(torch.ones(self.config.mult_k,
                                                        self.config.k)/config.k),
                                   requires_grad=True)
        self.register_parameter('log_py', self.log_py)

        self.log_uniform_y = Variable(torch.log(torch.ones(1) / config.k))
        if self.use_gpu:
            self.log_uniform_y = self.log_uniform_y.cuda()

        self.kl_w = 0.0

        self.return_latent_key = ("dec_init_state", "log_qy", "y_ids")

    @staticmethod
    def add_args(parser):
        from dgmvae.utils import str2bool
        # Latent variable:
        parser.add_argument('--k', type=int, default=5, help="Latent size of discrete latent variable")
        parser.add_argument('--mult_k', type=int, default=20, help="The number of discrete latent variables.")

        # Network setting:
        parser.add_argument('--rnn_cell', type=str, default='gru')
        parser.add_argument('--embed_size', type=int, default=512)
        parser.add_argument('--utt_type', type=str, default='rnn')
        parser.add_argument('--enc_cell_size', type=int, default=512)
        parser.add_argument('--dec_cell_size', type=int, default=512)
        parser.add_argument('--bi_enc_cell', type=str2bool, default=True)
        parser.add_argument('--num_layer_enc', type=int, default=1)
        parser.add_argument('--num_layer_dec', type=int, default=1)
        parser.add_argument('--use_attn', type=str2bool, default=False)
        parser.add_argument('--attn_type', type=str, default='cat')
        parser.add_argument('--tie_output_embed', type=str2bool, default=True)
        parser.add_argument('--max_utt_len', type=int, default=40)
        parser.add_argument('--max_dec_len', type=int, default=40)
        parser.add_argument('--max_vocab_cnt', type=int, default=10000)

        # Other settings:
        parser.add_argument('--use_mutual', type=str2bool, default=False)
        parser.add_argument('--use_kl', type=str2bool, default=True)
        parser.add_argument('--concat_decoder_input', type=str2bool, default=True)
        parser.add_argument('--gmm', type=str2bool, default=False)

        return parser

    def valid_loss(self, loss, batch_cnt=None, step=None):
        if batch_cnt is not None:
            step = batch_cnt

        if step is not None and 'anneal_function' in self.config:
            vae_kl_weight = kl_anneal_function(self.config.anneal_function, step,
                                               self.config.anneal_k, self.config.anneal_x0)
        else:
            vae_kl_weight = 1.0
        if self.config.use_mutual or self.config.anneal is not True:
            vae_kl_weight = 1.0

        total_loss = loss.nll

        if not self.use_kl:
            return total_loss

        if self.config.use_mutual:
            total_loss += (vae_kl_weight * loss.agg_ckl)
        else:
            total_loss += (vae_kl_weight * loss.ckl_real)

        return total_loss

    def model_sel_loss(self, loss, batch_cnt):
        if not self.use_kl:  # DAE
            return loss.nll
        else:
            if "sel_metric" in self.config and self.config.sel_metric == "elbo":
                return loss.elbo
            return self.valid_loss(loss)
            # return loss.elbo

    def forward(self, data_feed, mode, gen_type='greedy', sample_n=1, return_latent=False):
        posterior_sample_n = self.posterior_sample_n if self.training else 1

        if isinstance(data_feed, tuple):
            data_feed = data_feed[0]
        batch_size = len(data_feed['output_lens'])
        out_utts = self.np2var(data_feed['outputs'], LONG)

        # output encoder
        output_embedding = self.embedding(out_utts)
        x_outs, x_last = self.x_encoder(output_embedding)
        if type(x_last) is tuple:
            x_last = x_last[0].view(self.num_layer_enc, 1 + int(self.bi_enc_cell), -1, self.enc_cell_size)[-1]
            x_last = x_last.transpose(0, 1).contiguous().view(-1, self.enc_out_size)
        else:
            x_last = x_last.view(self.num_layer_enc, 1 + int(self.bi_enc_cell), -1, self.enc_cell_size)[-1]
            x_last = x_last.transpose(0, 1).contiguous().view(-1,
                                                              self.enc_out_size)


        # posterior network
        qy_logits = self.q_y(x_last).view(-1, self.config.k)
        log_qy = F.log_softmax(qy_logits, qy_logits.dim()-1)

        # switch that controls the sampling
        sample_y, y_ids = self.cat_connector(qy_logits.repeat(posterior_sample_n, 1),
                                             1.0, self.use_gpu, hard=not self.training, return_max_id=True)
        sample_y = sample_y.view(-1, self.config.k * self.config.mult_k)
        y_ids = y_ids.view(-1, self.config.mult_k)

        # map sample to initial state of decoder
        dec_init_state = self.dec_init_connector(sample_y)

        # get decoder inputs
        labels = out_utts[:, 1:].contiguous()
        dec_inputs = out_utts[:, 0:-1]

        # decode
        dec_outs, dec_last, dec_ctx = self.decoder(batch_size * posterior_sample_n,
                                                   dec_inputs.repeat(posterior_sample_n, 1),
                                                   dec_init_state,
                                                   mode=mode, gen_type=gen_type,
                                                   beam_size=self.beam_size,
                                                   latent_variable=sample_y)
        # compute loss or return results
        if mode == GEN:
            return dec_ctx, labels
        else:
            # RNN reconstruction
            nll = self.nll_loss(dec_outs, labels.repeat(posterior_sample_n, 1))
            if self.config.avg_type == "seq":
                ppl = self.ppl(dec_outs, labels.repeat(posterior_sample_n, 1))

            # regularization
            log_qy = log_qy.view(-1, self.config.mult_k, self.config.k)
            avg_log_qc = torch.log(torch.mean(torch.exp(log_qy), dim=0) + 1e-15)
            agg_ckl = self.cat_kl_loss(avg_log_qc, self.log_uniform_y, batch_size, unit_average=True, average=False)
            agg_ckl = torch.sum(agg_ckl)

            ckl_real = self.cat_kl_loss(log_qy, self.log_uniform_y, batch_size, unit_average=True, average=False)
            ckl_real = torch.sum(torch.mean(ckl_real.view(-1, self.config.k), dim=0))
            # H(C) - H(C|X)
            mi = - torch.sum(torch.exp(avg_log_qc) * avg_log_qc) + torch.sum(torch.exp(log_qy) * log_qy) / batch_size

            results = Pack(nll=nll, mi=mi, ckl_real=ckl_real,
                           elbo=nll+ckl_real, agg_ckl=agg_ckl)
            if self.config.avg_type == "seq":
                results['PPL'] = ppl

            if return_latent:
                results['log_qy'] = log_qy
                results['dec_init_state'] = dec_init_state
                results['y_ids'] = y_ids

            return results

    def sampling(self, batch_size):
        sample_y = torch.randint(0, self.config.k, [batch_size, self.config.mult_k], dtype=torch.long).cuda()
        cs = self.torch2var(idx2onehot(sample_y.view(-1), self.config.k)).view(-1, self.config.mult_k * self.config.k)
        dec_init_state = self.dec_init_connector(cs)

        _, _, outputs = self.decoder(cs.size(0),
                                      None, dec_init_state,
                                      mode=GEN, gen_type=self.config.gen_type,
                                      beam_size=self.config.beam_size,
                                     latent_variable=cs)
        return outputs

    def sampling_for_likelihood(self, batch_size, data_feed, sample_num, sample_type="LL"):
        # Importance sampling...
        # just for calculating log-likelihood
        if type(data_feed) is tuple:
            data_feed = data_feed[0]
        batch_size = len(data_feed['output_lens'])
        out_utts = self.np2var(data_feed['outputs'], LONG)  # batch_size * seq_len
        out_utts = out_utts.repeat(sample_num, 1)

        labels = out_utts[:, 1:].contiguous()
        dec_inputs = out_utts[:, 0:-1]

        output_embedding = self.embedding(out_utts)
        x_outs, x_last = self.x_encoder(output_embedding)
        if type(x_last) is tuple:
            x_last = x_last[0].view(self.num_layer_enc, 1 + int(self.bi_enc_cell), -1, self.enc_cell_size)[-1]
            x_last = x_last.transpose(0, 1).contiguous().view(-1, self.enc_out_size)
        else:
            x_last = x_last.view(self.num_layer_enc, 1 + int(self.bi_enc_cell), -1, self.enc_cell_size)[-1]
            x_last = x_last.transpose(0, 1).contiguous().view(-1,
                                                              self.enc_out_size)

        qy_logits = self.q_y(x_last).view(-1, self.config.k)
        log_qy = F.log_softmax(qy_logits, -1)
        sampling_c = torch.multinomial(torch.exp(log_qy), 1) # .view(-1, self.config.mult_k)  # [batch_size * mult_k, 1]
        log_qcx = torch.sum(torch.gather(log_qy, 1, sampling_c).view(-1, self.config.mult_k), dim=-1)
        sampling_c = self.torch2var(idx2onehot(sampling_c.view(-1), self.config.k)).view(-1, self.config.mult_k * self.config.k)

        # print(log_qcx.size())
        log_pc = math.log(1.0 / self.config.k) * self.config.mult_k

        # Calculate p(x|z)
        dec_init_state = self.dec_init_connector(sampling_c)
        dec_outs, dec_last, outputs = self.decoder(sampling_c.size(0),
                                                   dec_inputs, dec_init_state,
                                                   mode=TEACH_FORCE,
                                                   gen_type=self.config.gen_type,
                                                   beam_size=self.config.beam_size,
                                                   latent_variable=sampling_c if self.concat_decoder_input else None)

        # nll = self.nll_loss(dec_outs, labels)
        nll = F.nll_loss(dec_outs.view(-1, dec_outs.size(-1)), labels.view(-1), reduction="none").view(out_utts.size(0),
                                                                                                       -1)
        nll = torch.sum(nll, dim=-1)

        ll = torch.exp(-nll.double() + log_pc - log_qcx.double())  # log (p(z)p(x|z) / q(z|x))

        ll = ll.view(-1, sample_num)
        # nll_per = torch.log(torch.mean(ll, dim=-1))  #
        # batch_size = nll_per.size(0)
        # nll_per = torch.sum(nll_per)
        return ll

class GMVAE(BaseModel):
    def __init__(self, config):
        super(GMVAE, self).__init__(config)
        # 这一部分是针对语言处理的初始化，这些变量定义了词汇表的大小以及特殊标记

        # 网络参数
        self.embed_size = config.embed_size # 输入到模型中的神经元的数量
        self.decoder_output_size = config.latent_size # 解码器输出的大小
        self.num_layer_enc = config.num_layer_enc # 编码器（encoder）RNN 的层数。
        self.num_layer_dec = config.num_layer_dec # 解码器（decoder）RNN 的层数。
        self.dropout = config.dropout # RNN 层之间的 dropout 概率。
        self.enc_cell_size = config.enc_cell_size # 编码器 RNN 单元的隐藏层大小。
        self.dec_cell_size = config.dec_cell_size # 解码器 RNN 单元的隐藏层大小。
        self.rnn_cell = config.rnn_cell # RNN 单元的类型，可能是 GRU 或 LSTM。
        self.use_attn = config.use_attn # 是否使用注意力机制（attention）。
        self.beam_size = config.beam_size # 用于解码时的 beam search 大小。Beam Search 是一种启发式搜索算法，用于解码阶段选择最优的输出序列。
        self.utt_type = config.utt_type # 表示用于处理输入的单元类型。
        self.bi_enc_cell = config.bi_enc_cell # 是否使用双向 RNN 编码器。
        self.attn_type = config.attn_type # 注意力机制的类型。
        self.enc_out_size = self.enc_cell_size * 2 if self.bi_enc_cell else self.enc_cell_size # 编码器输出的大小。 如果是双向 RNN，则编码器的输出大小为 enc_cell_size * 2
        self.concat_decoder_input = config.concat_decoder_input if "concat_decoder_input" in config else False # 是否将潜变量和解码器输入进行拼接。 在解码器输入时，是否将潜在变量（例如变分自编码器的潜在空间向量）与输入词向量拼接，以增强生成能力。
        self.posterior_sample_n = config.post_sample_num if "post_sample_num" in config else 1 # 每次训练时从后验分布中采样的次数。


        # 编码器
        self.x_encoder = EncoderRNN(self.embed_size, self.enc_cell_size,
                                    dropout_p=self.dropout,
                                    rnn_cell=self.rnn_cell,
                                    variable_lengths=self.config.fix_batch,
                                    bidirection=self.bi_enc_cell,
                                    n_layers=self.num_layer_enc)
        # 解码器
        self.decoder = DecoderRNN(self.decoder_output_size, 
                                  input_size=self.dec_cell_size,
                                  hidden_size=self.dec_cell_size,
                                  n_layers=self.num_layer_dec, rnn_cell=self.rnn_cell,
                                  input_dropout_p=self.dropout,
                                  dropout_p=self.dropout,
                                  use_attention=self.use_attn,
                                  attn_size=self.enc_cell_size,
                                  attn_mode=self.attn_type,
                                  use_gpu=self.use_gpu)
        
        # 高斯混合模型 (GMM) 的设置
        # 潜变量的均值和对数方差
        self.q_y_mean = nn.Linear(self.enc_out_size, config.latent_size * config.mult_k) # config.mult_k， 用于扩展潜在变量空间的一个倍数，通常用于创建多个潜在分布或者捕捉更多的细粒度特征。
        self.q_y_logvar = nn.Linear(self.enc_out_size, config.latent_size * config.mult_k)

        self.post_c = nn.Sequential(
            nn.Linear(self.enc_out_size, self.enc_out_size),
            nn.ReLU(),
            nn.Linear(self.enc_out_size, self.config.mult_k * self.config.k), 
        ) # 对编码器输出的特征进行映射，以生成离散潜在变量的概率分布
        # 生成网络初始化连接器:该连接器将潜变量转换为解码器的初始状态，用于开始生成序列。
        # 它的作用是将从潜在空间中采样的潜在变量（latent variable）通过一个线性转换，映射到解码器隐藏状态的合适大小。一个线性变化
        self.dec_init_connector = nn_lib.LinearConnector(
            input_size=config.latent_size * config.mult_k,
            output_size=self.dec_cell_size,
            is_lstm = False,
            has_bias=False)
        self.firing_rate = nn.Linear(self.decoder_output_size, self.embed_size)
        # 这个对象 GumbelConnector 用于处理离散潜在变量的采样和重新参数化，尤其是在使用 Gumbel-Softmax 技术时。
        self.cat_connector = nn_lib.GumbelConnector() 
        # 这几行代码定义了模型中用于计算损失的三个损失函数，分别是 NLL（负对数似然）损失、类别 Kullback-Leibler（KL）散度损失 以及 困惑度（Perplexity）。
        # 这几个损失的计算也需要修改
        #self.mse_loss = criterions.TimeSeriesLossMSE()
        self.nll_loss = criterions.PoissonNLLLoss()
        self.cat_kl_loss = criterions.CatKLLoss()
        #self.ppl = criterions.Perplexity(self.config)

        self.init_gaussian()  # 初始化高斯分布
        # 定义了模型在生成或解码过程中，返回的一些潜在变量和状态的键名。
        self.return_latent_key = ('log_qy', 'dec_init_state', 'y_ids', 'z')
        self.kl_w = 0.0 # 表示KL 散度损失的权重，通常用于控制 KL 散度在总损失中的贡献。


    @staticmethod
    def add_args(parser):
        from dgmvae.utils import str2bool
        # Latent variable:
        parser.add_argument('--latent_size', type=int, default=2, help="The latent size of continuous latent variable.")
        parser.add_argument('--mult_k', type=int, default=1, help="The number of discrete latent variables.")
        parser.add_argument('--k', type=int, default=5, help="The dimension of discrete latent variable.")

        # Network setting:
        parser.add_argument('--rnn_cell', type=str, default='gru')
        parser.add_argument('--embed_size', type=int, default=100) # 输入到encoder的特征维度，针对于神经元数据，那就是神经元的数目
        parser.add_argument('--utt_type', type=str, default='rnn')
        parser.add_argument('--enc_cell_size', type=int, default=64)
        parser.add_argument('--dec_cell_size', type=int, default=64)
        parser.add_argument('--bi_enc_cell', type=str2bool, default=True)
        parser.add_argument('--num_layer_enc', type=int, default=1)
        parser.add_argument('--num_layer_dec', type=int, default=1)
        parser.add_argument('--use_attn', type=str2bool, default=False)
        parser.add_argument('--attn_type', type=str, default='cat')
        parser.add_argument('--tie_output_embed', type=str2bool, default=True)
        parser.add_argument('--max_vocab_cnt', type=int, default=10000)

        # Dispersed GMVAE settings:
        parser.add_argument('--use_mutual', type=str2bool, default=False)
        parser.add_argument('--beta', type=float, default=0.2)
        parser.add_argument('--concat_decoder_input', type=str2bool, default=True)
        parser.add_argument('--gmm', type=str2bool, default=True)
        parser.add_argument('--klw_for_ckl', type=float, default=1.0)
        parser.add_argument('--klw_for_zkl', type=float, default=1.0)
        parser.add_argument('--pretrain_ae_step', type=int, default=0)
        return parser
    # 初始化高斯分布
    def init_gaussian(self):
        self._log_uniform_y = Variable(torch.log(torch.ones(1) / self.config.k))
        if self.use_gpu:
            self._log_uniform_y = self.log_uniform_y.cuda()

        mus = torch.randn(self.config.mult_k, self.config.k, self.config.latent_size)
        logvar = torch.randn(self.config.mult_k, self.config.k, self.config.latent_size)
        if torch.cuda.is_available():
            mus = mus.cuda()
            logvar = logvar.cuda()
        self._gaussian_mus = torch.nn.Parameter(mus, requires_grad=True)  # change: False
        self._gaussian_logvar = torch.nn.Parameter(logvar, requires_grad=True)  # change: False

    @property
    def gaussian_mus(self):
        return self._gaussian_mus

    @property
    def gaussian_logvar(self):
        return self._gaussian_logvar

    @property
    def log_uniform_y(self):
        return self._log_uniform_y
    # 用于选择模型在训练过程中的损失值，以决定模型参数的更新方向。这段代码的作用是在训练过程中，根据特定条件从不同的损失指标中选择一个用于优化的损失值。
    def model_sel_loss(self, loss, batch_cnt):
        # 在预训练阶段，函数返回 负对数似然损失 (loss.nll)。这是因为在预训练阶段，模型可能主要集中在自编码的重构能力上，不涉及复杂的潜在空间优化。因此，优化 NLL（Negative Log-Likelihood）重构误差可以帮助模型更好地学习如何重构输入。
        if batch_cnt is not None and batch_cnt < self.config.pretrain_ae_step:
            return loss.nll
        # 这通常用于模型已经具备较好的重构能力后，进一步优化潜在空间的结构，以便更好地生成合理的潜在表示。
        if "sel_metric" in self.config and self.config.sel_metric == "elbo":
            return loss.elbo
        # 如果没有特别指定，就使用 valid_loss，它结合了重构和 KL 散度的损失。
        return self.valid_loss(loss)
    # 冻结模型中与识别网络（recognition network）有关的参数，使得这些参数在训练过程中不再更新。
    def freeze_recognition_net(self):
        for param in self.embedding.parameters():
            param.requires_grad = False
        for param in self.x_encoder.parameters():
            param.requires_grad = False
        for param in self.q_y_mean.parameters():
            param.requires_grad = False
        for param in self.q_y_logvar.parameters():
            param.requires_grad = False
        for param in self.post_c.parameters():
            param.requires_grad = False
        for param in self.dec_init_connector.parameters():
            param.requires_grad = False

    def freeze_generation_net(self):
        for param in self.decoder.parameters():
            param.requires_grad = False
        self.gaussian_mus.requires_grad = False
        self.gaussian_logvar.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True
    # 计算模型的验证损失，这个损失由多个部分组成，用于衡量模型在当前阶段的表现情况。
    def valid_loss(self, loss, batch_cnt=None, step=None):
        if batch_cnt is not None:
            step = batch_cnt

        if batch_cnt is not None and batch_cnt < self.config.pretrain_ae_step:
            return loss.nll
        if step == self.config.pretrain_ae_step:
            self.flush_valid = True
        # 在变分自编码器（VAE）中，KL 散度退火（KL Annealing） 是一种策略，用于逐步增加 KL 散度的权重，使模型在早期更专注于最小化重构误差，从而更好地学习重构能力，然后逐渐将注意力转移到潜在空间的正则化上。
        if step is not None and 'anneal_function' in self.config:
            vae_kl_weight = kl_anneal_function(self.config.anneal_function, step - self.config.pretrain_ae_step,
                                               self.config.anneal_k, self.config.anneal_x0,
                                               self.config.anneal_warm_up_step if "anneal_warm_up_step" in self.config else 0,
                                               self.config.anneal_warm_up_step if "anneal_warm_up_value" in self.config else 0)
        else:
            vae_kl_weight = 1.0

        if not self.config.anneal:
            vae_kl_weight = 1.0

        mi_weight = 0.0 if self.config.use_mutual else 1.0

        total_loss = loss.nll + vae_kl_weight * (self.config.klw_for_ckl * (loss.agg_ckl + mi_weight * loss.mi) +
                                                 self.config.klw_for_zkl * (loss.zkl + self.config.beta * loss.dispersion)
                                                )
        return total_loss
    # 这个函数 reparameterization 的作用是实现重参数化技巧（reparameterization trick），用于从高斯分布中采样潜在变量 z，以便使得整个采样过程是可微的，从而可以通过反向传播训练模型。这种技巧通常用于变分自编码器（VAE）中，以实现潜在变量的可微采样。
    def reparameterization(self, mu, logvar, sample=True):
        if self.training or sample:
            std = torch.exp(0.5 * logvar)
            z = self.torch2var(torch.randn(mu.size()))
            z = z * std + mu
            return z
        else:
            return mu
    # zkl_loss 函数计算了模型的潜在变量的 KL 散度损失，特别是对潜在空间中每个潜在变量的均值和方差进行对比，从而衡量模型生成的后验分布和给定的先验分布之间的差异。
    def zkl_loss(self, tgt_probs, mean, log_var, mean_prior=True):
        # tgt_probs: 目标概率（可能是标签或某种概率分布）。它的形状为 [batch_size, mult_k, latent_size]，表示每个样本在每个潜在维度上的概率。
        # mean: 变分后验的均值，形状为 [batch_size, mult_k, latent_size]。
        # log_var: 变分后验的对数方差，形状与 mean 相同。
        # mean_prior: 一个布尔值，如果为 True，使用先验的均值；如果为 False，使用固定的高斯分布。
        mean = mean.view(-1, self.config.mult_k, self.config.latent_size) #？？为什么最后的维度是latnet_size
        log_var = log_var.view(-1, self.config.mult_k, self.config.latent_size)
        if mean_prior:
            tgt_probs_ = tgt_probs.unsqueeze(-1).expand(-1, -1, -1, self.config.latent_size)
            eta1 = self.gaussian_mus / torch.exp(self.gaussian_logvar)  # eta1 = \Sigma^-1 * mu
            eta2 = -0.5 * torch.pow(torch.exp(self.gaussian_logvar), -1)
            Eeta1 = torch.sum(tgt_probs_ * eta1, dim=-2)  # [batch_size, mult_k, latent_size]
            Eeta2 = torch.sum(tgt_probs_ * eta2, dim=-2)
            Emu = -0.5 * Eeta1 / Eeta2
            Evar = -0.5 * torch.pow(Eeta2, -1)
            # [batch_size, mult_k, latent_size]
            kl = 0.5 * (
                    torch.sum(log_var.exp().div(Evar), dim=-1)
                    + torch.sum((Emu - mean).pow(2) / Evar, dim=-1)
                    - mean.size(-1)
                    + torch.sum(Evar.log() - log_var, dim=-1)
            )
            # [batch_size, mult_k]
            return kl

        mu_repeat = mean.unsqueeze(-2).expand(-1, -1, self.config.k, -1)  # batch_size x k x z_dim
        logvar_repeat = log_var.unsqueeze(-2).expand(-1, -1, self.config.k, -1)
        gaussian_logvars = self.gaussian_logvar

        kl = 0.5 * (
                torch.sum(logvar_repeat.exp().div(gaussian_logvars.exp()), dim=-1)
                + torch.sum((self.gaussian_mus - mu_repeat).pow(2) / gaussian_logvars.exp(), dim=-1)
                - mean.size(-1)
                + torch.sum((gaussian_logvars - logvar_repeat), dim=-1)
        )  # batch_size x mult_k x k

        return torch.sum(kl * tgt_probs, dim=-1)  # batch_size*mult_k
    # 该函数的核心目标是衡量模型生成的潜在变量在各类别之间的差异性，进而鼓励模型在潜在空间中形成良好且具有多样性的分布结构。
    def dispersion(self, tgt_probs):
        # tgt_probs: batch_size x mult_k x k
        tgt_probs_ = tgt_probs.unsqueeze(-1).expand(-1, -1, -1, self.config.latent_size)
        eta1 = self.gaussian_mus / torch.exp(self.gaussian_logvar)  # eta1 = \Sigma^-1 * mu
        eta2 = -0.5 * torch.pow(torch.exp(self.gaussian_logvar), -1)
        Eeta1 = torch.sum(tgt_probs_ * eta1, dim=-2) # [batch_size, mult_k, latent_size]
        Eeta2 = torch.sum(tgt_probs_ * eta2, dim=-2)
        AE = -0.25 * Eeta1 * Eeta1 / Eeta2 - 0.5 * torch.log(-2 * Eeta2) # [batch_size, mult_k, latent_size]
        AE = torch.mean(torch.sum(AE, dim=(-1, -2)))

        EA = torch.sum(-0.25 * eta1 * eta1 / eta2 - 0.5 * torch.log(-2 * eta2), dim=-1) # [mult_k, k]
        EA = torch.mean(torch.sum(tgt_probs * EA, dim=(-1,-2)))
        return EA-AE
    # 计算自然参数（natural parameters）的加权方差，用于评估潜在空间的参数变化程度。
    def param_var(self, tgt_probs):
        # Weighted variance of natural parameters
        # tgt_probs: batch_size x mult_k x k
        tgt_probs_ = tgt_probs.unsqueeze(-1).expand(-1, -1, -1, self.config.latent_size)
        eta1 = self.gaussian_mus / torch.exp(self.gaussian_logvar)  # eta1 = \Sigma^-1 * mu
        eta2 = -0.5 * torch.pow(torch.exp(self.gaussian_logvar), -1)

        var_eta1 = torch.sum(tgt_probs_ * (eta1 * eta1), dim=-2) - torch.sum(tgt_probs_ * eta1, dim=-2).pow(2)
        var_eta2 = torch.sum(tgt_probs_ * (eta2 * eta2), dim=-2) - torch.sum(tgt_probs_ * eta2, dim=-2).pow(2)

        return torch.sum(var_eta1 + var_eta2) / tgt_probs.size(0)
    # forward 方法定义了数据的前向传播过程，包括编码、从潜在空间中采样、解码等步骤
    def forward(self, data_feed, mode, gen_type='greedy', sample_n=1, return_latent=False):
        # 如果模型处于训练状态 (self.training 为 True)，则使用预设的采样数量 self.posterior_sample_n。否则，即在评估或生成状态下，采样数量为 1。
        posterior_sample_n = self.posterior_sample_n if self.training else 1

        # 检查并确保 data_feed 不是一个元组，将其转换为字典形式。
        if type(data_feed['x']) is tuple:
            data_feed['x'] = data_feed['x'][0]
        batch_size = len(data_feed['x'])
        # 这行代码将输入数据（通常是 NumPy 数组）转换为 PyTorch 张量，并且根据需要将其移动到 GPU 上。这使得该张量可以被模型直接用于训练或推理。
        batch_spikes = self.np2var(data_feed['x'], FLOAT)
        batch_seq_len = batch_spikes.size(1)
        # 动态更新 max_dec_len
        max_dec_len = batch_seq_len       

        
        """
        # output encoder
        output_embedding = self.embedding(out_utts) 
        """
        # 编码器将嵌入表示进行编码，生成编码输出 (x_outs) 和最后一个隐藏状态 (x_last)，用于后续的潜在变量计算。
        x_outs, x_last = self.x_encoder(batch_spikes)
        # 对 x_last 进行形状变换，以匹配潜在变量的维度需求。
        # 最终，x_last 被展平并转换为 [batch_size, enc_out_size]。
        if type(x_last) is tuple:
            x_last = x_last[0].view(self.num_layer_enc, 1 + int(self.bi_enc_cell), -1, self.enc_cell_size)[-1]
            x_last = x_last.transpose(0, 1).contiguous().view(-1, self.enc_out_size)
        else:
            x_last = x_last.view(self.num_layer_enc, 1 + int(self.bi_enc_cell), -1, self.enc_cell_size)[-1]
            x_last = x_last.transpose(0, 1).contiguous().view(-1,
                                                              self.enc_out_size)
        # 计算潜在变量 z 和离散变量 c 
        # q(z|x)
        qz_mean = self.q_y_mean(x_last)  # batch x (latent_size*mult_k)
        qz_logvar = self.q_y_logvar(x_last)
        # 重参数化之后再计算潜在变量 z
        sample_z = self.reparameterization(qz_mean.repeat(posterior_sample_n, 1),
                                      qz_logvar.repeat(posterior_sample_n, 1),
                                      sample=gen_type != "greedy" or mode != GEN)  # batch x (latent_size*mult_k)
        # 这个地方直接使用z的均值
        sample_z_mean = qz_mean
        # q(c|x)
        qc_logits = self.post_c(x_last).view(-1, self.config.k)
        log_qc = F.log_softmax(qc_logits, -1)  # [batch*mult_k, k]
        # cat_connector 可能会基于输入的 logits（qc_logits），根据是否训练选择软采样（soft sampling）或硬采样（hard sampling），并返回一个类别采样结果 sample_c 和类别索引 c_ids。
        sample_c, c_ids = self.cat_connector(qc_logits, 1.0, self.use_gpu, hard=not self.training, return_max_id=True)
        # sample_c: [batch*mult_k, k], c_ids: [batch*mult_k, 1]
        # sample_c = sample_c.view(-1, self.config.mult_k * self.config.k)
        c_ids = c_ids.view(-1, self.config.mult_k)

        # Prepare for decoding 准备解码输入
        # 这个地方潜变量的维度应该是latent_size 
        dec_init_state = self.dec_init_connector(sample_z_mean)
        # 分别为目标序列（去掉第一个标记）和解码器的输入（去掉最后一个标记）。这个地方是teacher forcing操作需要的
        #labels = out_utts[:, 1:].contiguous()
        #dec_inputs = out_utts[:, 0:-1]
        # 这里是解码器的输入，这里是一个全零的张量，用于初始化解码器的输入。
        # 这个地方不需要初始化输入，在decoder中有处理
        #dec_inputs = torch.zeros(batch_size, self.max_dec_len, dtype=torch.long).cuda() if self.use_gpu else torch.zeros(batch_size, self.max_dec_len, dtype=torch.long)
        # decode 解码器部分，参考LFADS的解码器部分
        dec_outs, dec_last = self.decoder(max_dec_len, batch_size, inputs=None, init_state=dec_init_state,
                                                   latent_variable=sample_z if self.concat_decoder_input else None)

        
        # 一层MLP+exp转化成firing rate
        for_firing_rate = self.firing_rate(dec_outs)
        firing_rate = torch.exp(for_firing_rate) # Poisson分布的均值，lamda
        # 这个地方需要生成spike
        output_spikes = torch.tensor(np.random.poisson(firing_rate.detach().cpu().numpy())).to('cuda')



        # labels可以直接从输入数据中获取
        # compute loss or return results
        labels = batch_spikes

        # RNN reconstruction
        # 这是计算解码器输出与目标标签之间的负对数似然损失（Negative Log-Likelihood）。dec_outs 是解码器的输出，labels 是目标标签。
        nll = self.nll_loss(firing_rate, labels)
        #mse = self.mse_loss(output_spikes, labels)
        # 困惑度（Perplexity），它是用来评估语言模型的指标。它与 NLL 紧密相关，并且是训练模型时的一项常用评估指标。
        #ppl = self.ppl(dec_outs, labels)
        # Regularization terms
        # 这是计算类别的概率分布的步骤。
        qc = torch.exp(log_qc.view(-1, self.config.mult_k, self.config.k))
        # ZKL & dispersion term
        #  这是计算潜在类别变量分布的离散度。这个项通常用于正则化，旨在避免类别分布过于集中的情况。
        zkl = self.zkl_loss(qc, qz_mean, qz_logvar, mean_prior=True)  # [batch_size x mult_k]
        zkl_real = self.zkl_loss(qc, qz_mean, qz_logvar, mean_prior=False)  # [batch_size x mult_k]
        zkl = torch.sum(torch.mean(zkl, dim=0))
        zkl_real = torch.sum(torch.mean(zkl_real, dim=0))
        dispersion = self.dispersion(qc)
        # CKL & MI term
        # agg_ckl: 这是类别的 KL 散度（Categorical KL divergence），用于度量类别分布与均匀分布之间的差异。
        # 这个地方的维度是什么
        avg_log_qc = torch.log(torch.mean(qc, dim=0) + 1e-15)
        agg_ckl = self.cat_kl_loss(avg_log_qc, self.log_uniform_y, batch_size, unit_average=True, average=False)
        agg_ckl = torch.sum(agg_ckl)
        # 这是计算类别的实际 KL 散度。
        ckl_real = self.cat_kl_loss(log_qc, self.log_uniform_y, batch_size, unit_average=True, average=False)
        ckl_real = torch.sum(torch.mean(ckl_real.view(-1, self.config.mult_k), dim=0))
        # H(C) - H(C|X)
        # 互信息（Mutual Information），衡量类别变量和其他潜变量（如 qc）之间的信息量。它用来衡量模型对潜变量的约束和不确定性的控制。
        mi = - torch.sum(torch.exp(avg_log_qc) * avg_log_qc) + torch.sum(torch.exp(log_qc) * log_qc) / batch_size
        # 使用 Pack 类组织模型的各种损失项和潜在变量度量。
        results = Pack(nll=nll, agg_ckl=agg_ckl, mi=mi, zkl=zkl, dispersion=dispersion,
                        real_zkl=zkl_real, real_ckl=ckl_real, elbo=nll + ckl_real + zkl_real,
                        param_var=self.param_var(tgt_probs=qc))

        if return_latent:
            results['log_qy'] = log_qc
            results['dec_init_state'] = dec_init_state
            results['y_ids'] = c_ids
            results['z'] = sample_z

        return results
    # 使用了重要性采样的方法来估计生成模型的对数似然（log-likelihood）。
    def sampling_for_likelihood(self, batch_size, data_feed, sample_num, sample_type="LL",
                                ):
        # Importance sampling for estimating the log-likelihood
        assert sample_type in ("LL", "logLL")

        if type(data_feed) is tuple:
            data_feed = data_feed[0]
        batch_size = len(data_feed['output_lens'])
        out_utts = self.np2var(data_feed['outputs'], LONG)  # batch_size * seq_len
        out_utts = out_utts.repeat(sample_num, 1)

        labels = out_utts[:, 1:].contiguous()
        dec_inputs = out_utts[:, 0:-1]

        output_embedding = self.embedding(out_utts)
        x_outs, x_last = self.x_encoder(output_embedding)
        if type(x_last) is tuple:
            x_last = x_last[0].view(self.num_layer_enc, 1 + int(self.bi_enc_cell), -1, self.enc_cell_size)[-1]
            x_last = x_last.transpose(0, 1).contiguous().view(-1, self.enc_out_size)
        else:
            x_last = x_last.view(self.num_layer_enc, 1 + int(self.bi_enc_cell), -1, self.enc_cell_size)[-1]
            x_last = x_last.transpose(0, 1).contiguous().view(-1,
                                                              self.enc_out_size)

        # q(z|x)
        qz_mean = self.q_y_mean(x_last)  # batch x (latent_size*mult_k)
        qz_logvar = self.q_y_logvar(x_last)
        sample_z = self.reparameterization(qz_mean, qz_logvar, sample=True)

        log_qzx = torch.sum(
            - (sample_z - qz_mean) * (sample_z - qz_mean) / (2 * torch.exp(qz_logvar)) - 0.5 * qz_logvar - 0.5 * math.log(
                math.pi * 2),
            dim=-1)
        sample_z_repeat = sample_z.view(-1, self.config.mult_k, 1, self.config.latent_size).repeat(1, 1, self.config.k, 1)
        log_pzc = torch.sum(
            - (sample_z_repeat - self.gaussian_mus) * (sample_z_repeat - self.gaussian_mus) / (2 * torch.exp(self.gaussian_logvar))
            - 0.5 * self.gaussian_logvar - 0.5 * math.log(math.pi * 2),
            dim=-1)  # [batch_size, mult_k, k]
        log_pz = torch.log(torch.mean(torch.exp(log_pzc.double()), dim=-1))  #
        log_pz = torch.sum(log_pz, dim=-1)

        # Calculate p(x|z)
        dec_init_state = self.dec_init_connector(sample_z)
        dec_outs, dec_last, outputs = self.decoder(sample_z.size(0),
                                                   dec_inputs, dec_init_state,
                                                   mode=TEACH_FORCE,
                                                   gen_type=self.config.gen_type,
                                                   beam_size=self.config.beam_size,
                                                   latent_variable=sample_z if self.concat_decoder_input else None)

        nll = F.nll_loss(dec_outs.view(-1, dec_outs.size(-1)), labels.view(-1), reduction="none").view(out_utts.size(0),
                                                                                                       -1)
        nll = torch.sum(nll, dim=-1)
        if sample_type == "logLL":
            return (-nll.double() + log_pz - log_qzx.double()).view(-1, sample_num)
        else:
            ll = torch.exp(-nll.double() + log_pz - log_qzx.double())  # exp ( log (p(z)p(x|z) / q(z|x)) )
            ll = ll.view(-1, sample_num)
        return ll
    # 这个函数 sampling 是用于从模型的先验分布中采样潜在变量，并使用这些采样的潜在变量生成新数据（通过解码器），实现生成模型的采样功能。它可以用于从训练好的模型中生成新的数据样本，从而展示模型的生成能力。
    # 这个函数的执行流程是：首先从先验分布中采样离散和连续的潜在变量，然后使用这些潜在变量生成解码器的初始状态，再通过解码器生成输出序列。
    def sampling(self, batch_size):
        sample_c = torch.randint(0, self.config.k, [batch_size, self.config.mult_k], dtype=torch.long).cuda()
        index = (self.torch2var(torch.arange(self.config.mult_k) * self.config.k) + sample_c).view(-1)
        mean = self.gaussian_mus.view(-1, self.config.latent_size)[index].squeeze()
        sigma = torch.exp(self.gaussian_logvar * 0.5).view(-1, self.config.latent_size)[index].squeeze()
        zs = self.reparameterization(mean, 2 * torch.log(torch.abs(sigma) + 1e-15), sample=True)
        zs = zs.view(-1, self.config.mult_k * self.config.latent_size)
        dec_init_state = self.dec_init_connector(zs)
        _, _, outputs = self.decoder(zs.size(0),
                                     None, dec_init_state,
                                     mode=GEN, gen_type="greedy",
                                     beam_size=self.config.beam_size,
                                     latent_variable=zs if self.concat_decoder_input else None)
        return outputs

class GMVAE_fb(GMVAE):
    @staticmethod
    def add_args(parser):
        from dgmvae.utils import str2bool
        # Latent variable:
        parser.add_argument('--latent_size', type=int, default=2, help="The latent size of continuous latent variable.")
        parser.add_argument('--mult_k', type=int, default=20, help="The number of discrete latent variables.")
        parser.add_argument('--k', type=int, default=5, help="The dimension of discrete latent variable.")

        # Network setting:
        parser.add_argument('--rnn_cell', type=str, default='gru')
        parser.add_argument('--embed_size', type=int, default=512)
        parser.add_argument('--utt_type', type=str, default='rnn')
        parser.add_argument('--enc_cell_size', type=int, default=512)
        parser.add_argument('--dec_cell_size', type=int, default=512)
        parser.add_argument('--bi_enc_cell', type=str2bool, default=True)
        parser.add_argument('--num_layer_enc', type=int, default=1)
        parser.add_argument('--num_layer_dec', type=int, default=1)
        parser.add_argument('--use_attn', type=str2bool, default=False)
        parser.add_argument('--attn_type', type=str, default='cat')
        parser.add_argument('--tie_output_embed', type=str2bool, default=True)
        parser.add_argument('--max_utt_len', type=int, default=40)
        parser.add_argument('--max_dec_len', type=int, default=40)
        parser.add_argument('--max_vocab_cnt', type=int, default=10000)

        # Dispersed GMVAE settings:
        parser.add_argument('--use_mutual', type=str2bool, default=False)
        parser.add_argument('--beta', type=float, default=0.2)
        parser.add_argument('--concat_decoder_input', type=str2bool, default=True)
        parser.add_argument('--gmm', type=str2bool, default=True)
        parser.add_argument('--pretrain_ae_step', type=int, default=0)

        # Free bits setting:
        parser.add_argument('--max_fb_c', type=float, default=5.0)
        parser.add_argument('--max_fb_z', type=float, default=10.0)

        return parser

    def model_sel_loss(self, loss, batch_cnt):  # return albo
        if batch_cnt is not None and batch_cnt < self.config.pretrain_ae_step:
            return loss.nll
        if "sel_metric" in self.config and self.config.sel_metric == "elbo":
            return loss.elbo
        return self.valid_loss(loss)

    def valid_loss(self, loss, batch_cnt=None, step=None):
        if batch_cnt is not None:
            step = batch_cnt

        if step < self.config.pretrain_ae_step:
            return loss.nll  # AE
        if step == self.config.pretrain_ae_step:
            self.flush_valid = True

        if step is not None and 'anneal_function' in self.config:
            vae_kl_weight = kl_anneal_function(self.config.anneal_function, step - self.config.pretrain_ae_step,
                                               self.config.anneal_k, self.config.anneal_x0,
                                               self.config.anneal_warm_up_step if "anneal_warm_up_step" in self.config else 0,
                                               self.config.anneal_warm_up_value if "anneal_warm_up_value" in self.config else 0)
        else:
            vae_kl_weight = 1.0

        if not self.config.anneal:
            vae_kl_weight = 1.0

        total_loss = loss.nll + vae_kl_weight * (loss.agg_ckl + loss.zkl)

        return total_loss

    def forward(self, data_feed, mode, gen_type='greedy', sample_n=1, return_latent=False):
        posterior_sample_n = self.posterior_sample_n if self.training else 1

        if type(data_feed) is tuple:
            data_feed = data_feed[0]
        batch_size = len(data_feed['output_lens'])
        out_utts = self.np2var(data_feed['outputs'], LONG)

        # output encoder
        output_embedding = self.embedding(out_utts)
        x_outs, x_last = self.x_encoder(output_embedding)
        if type(x_last) is tuple:
            x_last = x_last[0].view(self.num_layer_enc, 1 + int(self.bi_enc_cell), -1, self.enc_cell_size)[-1]
            x_last = x_last.transpose(0, 1).contiguous().view(-1, self.enc_out_size)
        else:
            x_last = x_last.view(self.num_layer_enc, 1 + int(self.bi_enc_cell), -1, self.enc_cell_size)[-1]
            x_last = x_last.transpose(0, 1).contiguous().view(-1,
                                                              self.enc_out_size)

        # q(z|x)
        qz_mean = self.q_y_mean(x_last)  # batch x (latent_size*mult_k)
        qz_logvar = self.q_y_logvar(x_last)
        sample_z = self.reparameterization(qz_mean.repeat(posterior_sample_n, 1),
                                           qz_logvar.repeat(posterior_sample_n, 1),
                                           sample=gen_type != "greedy" or mode != GEN)  # batch x (latent_size*mult_k)
        # q(c|x)
        qc_logits = self.post_c(x_last).view(-1, self.config.k)
        log_qc = F.log_softmax(qc_logits, -1)  # [batch*mult_k, k]
        sample_c, c_ids = self.cat_connector(qc_logits, 1.0, self.use_gpu, hard=not self.training, return_max_id=True)
        # sample_c: [batch*mult_k, k], c_ids: [batch*mult_k, 1]
        # sample_c = sample_c.view(-1, self.config.mult_k * self.config.k)
        c_ids = c_ids.view(-1, self.config.mult_k)

        # Prepare for decoding
        dec_init_state = self.dec_init_connector(sample_z)
        labels = out_utts[:, 1:].contiguous()
        dec_inputs = out_utts[:, 0:-1]
        if self.config.word_dropout_rate > 0:
            # randomly replace decoder input with <unk>
            prob = torch.rand(dec_inputs.size())
            prob[(dec_inputs.data - self.go_id) * (dec_inputs.data - self.pad_id) * (
                        dec_inputs.data - self.eos_id) == 0] = 1
            dec_inputs_copy = dec_inputs.clone()
            dec_inputs_copy[prob < self.config.word_dropout_rate] = self.unk_id
            dec_inputs = dec_inputs_copy

        # decode
        dec_outs, dec_last, dec_ctx = self.decoder(batch_size * posterior_sample_n,
                                                   dec_inputs.repeat(posterior_sample_n, 1),
                                                   dec_init_state,
                                                   mode=mode, gen_type=gen_type,
                                                   beam_size=self.beam_size,
                                                   latent_variable=sample_z if self.concat_decoder_input else None)

        # compute loss or return results
        if mode == GEN:
            return dec_ctx, labels
        else:
            # RNN reconstruction
            nll = self.nll_loss(dec_outs, labels.repeat(posterior_sample_n, 1))
            ppl = self.ppl(dec_outs, labels.repeat(posterior_sample_n, 1))
            # Regularization terms
            qc = torch.exp(log_qc.view(-1, self.config.mult_k, self.config.k))
            # ZKL & dispersion term
            # zkl = self.zkl_loss(qc, qz_mean, qz_logvar, mean_prior=True)  # [batch_size x mult_k]
            zkl_real = self.zkl_loss(qc, qz_mean, qz_logvar, mean_prior=False)  # [batch_size x mult_k]
            zkl = torch.gt(zkl_real, self.config.max_fb_z / self.config.mult_k).float() * zkl_real
            zkl = torch.sum(torch.mean(zkl, dim=0))
            zkl_real = torch.sum(torch.mean(zkl_real, dim=0))
            dispersion = self.dispersion(qc)
            # CKL & MI term
            avg_log_qc = torch.log(torch.mean(qc, dim=0) + 1e-15)
            # agg_ckl = self.cat_kl_loss(avg_log_qc, self.log_uniform_y, batch_size, unit_average=True, average=False)
            # agg_ckl = torch.sum(agg_ckl)
            ckl_real = self.cat_kl_loss(log_qc, self.log_uniform_y, batch_size, unit_average=True, average=False)
            agg_ckl = torch.gt(ckl_real, self.config.max_fb_c / self.config.mult_k).float() * ckl_real
            ckl_real = torch.sum(torch.mean(ckl_real.view(-1, self.config.mult_k), dim=0))
            agg_ckl = torch.sum(torch.mean(agg_ckl.view(-1, self.config.mult_k), dim=0))
            # H(C) - H(C|X)
            mi = - torch.sum(torch.exp(avg_log_qc) * avg_log_qc) + torch.sum(torch.exp(log_qc) * log_qc) / batch_size

            results = Pack(nll=nll, agg_ckl=agg_ckl, mi=mi, zkl=zkl, dispersion=dispersion, PPL=ppl,
                           real_zkl=zkl_real, real_ckl=ckl_real, elbo=nll + ckl_real + zkl_real,
                           param_var=self.param_var(tgt_probs=qc))

            if return_latent:
                results['log_qy'] = log_qc
                results['dec_init_state'] = dec_init_state
                results['y_ids'] = c_ids
                results['z'] = sample_z

            return results

class GMVAE_MoP(BaseModel):
    def __init__(self, corpus, config):
        super(GMVAE_MoP, self).__init__(config)
        self.vocab = corpus.vocab
        self.rev_vocab = corpus.rev_vocab
        self.vocab_size = len(self.vocab)
        self.embed_size = config.embed_size
        self.max_utt_len = config.max_utt_len
        self.go_id = self.rev_vocab[BOS]
        self.eos_id = self.rev_vocab[EOS]
        self.unk_id = self.rev_vocab[corpus.unk]
        self.pad_id = self.rev_vocab[PAD]
        self.num_layer_enc = config.num_layer_enc
        self.num_layer_dec = config.num_layer_dec
        self.dropout = config.dropout
        self.enc_cell_size = config.enc_cell_size
        self.dec_cell_size = config.dec_cell_size
        self.rnn_cell = config.rnn_cell
        self.max_dec_len = config.max_dec_len
        self.use_attn = config.use_attn
        self.beam_size = config.beam_size
        self.utt_type = config.utt_type
        self.bi_enc_cell = config.bi_enc_cell
        self.attn_type = config.attn_type
        self.enc_out_size = self.enc_cell_size * 2 if self.bi_enc_cell else self.enc_cell_size
        self.concat_decoder_input = config.concat_decoder_input if "concat_decoder_input" in config else False
        self.posterior_sample_n = config.post_sample_num if "post_sample_num" in config else 1

        # build model here
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size,
                                      padding_idx=self.rev_vocab[PAD])
        self.x_encoder = EncoderRNN(self.embed_size, self.enc_cell_size,
                                    dropout_p=self.dropout,
                                    rnn_cell=self.rnn_cell,
                                    variable_lengths=self.config.fix_batch,
                                    bidirection=self.bi_enc_cell,
                                    n_layers=self.num_layer_enc)
        self.decoder = DecoderRNN(self.vocab_size, self.max_dec_len,
                                  self.embed_size + self.config.mult_k * self.config.latent_size if self.concat_decoder_input else self.embed_size,
                                  self.dec_cell_size,
                                  self.go_id, self.eos_id, self.unk_id,
                                  n_layers=self.num_layer_dec, rnn_cell=self.rnn_cell,
                                  input_dropout_p=self.dropout,
                                  dropout_p=self.dropout,
                                  use_attention=self.use_attn,
                                  attn_size=self.enc_cell_size,
                                  attn_mode=self.attn_type,
                                  use_gpu=self.use_gpu,
                                  tie_output_embed=config.tie_output_embed,
                                  embedding=self.embedding)
        self.q_y_mean = nn.Linear(self.enc_out_size, config.latent_size * config.mult_k)
        self.q_y_logvar = nn.Linear(self.enc_out_size, config.latent_size * config.mult_k)
        self.post_c = nn.Sequential(
            nn.Linear(self.enc_out_size, self.enc_out_size),
            nn.ReLU(),
            nn.Linear(self.enc_out_size, self.config.mult_k * self.config.k),
        )
        self.dec_init_connector = nn_lib.LinearConnector(
            config.latent_size * config.mult_k,
            self.dec_cell_size,
            self.rnn_cell == 'lstm',
            has_bias=False)
        self.cat_connector = nn_lib.GumbelConnector()

        self.nll_loss = criterions.NLLEntropy(self.rev_vocab[PAD], self.config)
        self.cat_kl_loss = criterions.CatKLLoss()
        self.ppl = criterions.Perplexity(self.rev_vocab[PAD], self.config)

        self.init_gaussian()

        self.return_latent_key = ('log_qy', 'dec_init_state', 'y_ids', 'z')
        self.kl_w = 0.0

    @staticmethod
    def add_args(parser):
        from dgmvae.utils import str2bool
        # Latent variable:
        parser.add_argument('--latent_size', type=int, default=2, help="The latent size of continuous latent variable.")
        parser.add_argument('--mult_k', type=int, default=20, help="The number of discrete latent variables.")
        parser.add_argument('--k', type=int, default=5, help="The dimension of discrete latent variable.")

        # Network setting:
        parser.add_argument('--rnn_cell', type=str, default='gru')
        parser.add_argument('--embed_size', type=int, default=512)
        parser.add_argument('--utt_type', type=str, default='rnn')
        parser.add_argument('--enc_cell_size', type=int, default=512)
        parser.add_argument('--dec_cell_size', type=int, default=512)
        parser.add_argument('--bi_enc_cell', type=str2bool, default=True)
        parser.add_argument('--num_layer_enc', type=int, default=1)
        parser.add_argument('--num_layer_dec', type=int, default=1)
        parser.add_argument('--use_attn', type=str2bool, default=False)
        parser.add_argument('--attn_type', type=str, default='cat')
        parser.add_argument('--tie_output_embed', type=str2bool, default=True)
        parser.add_argument('--max_utt_len', type=int, default=40)
        parser.add_argument('--max_dec_len', type=int, default=40)
        parser.add_argument('--max_vocab_cnt', type=int, default=10000)

        # Dispersed GMVAE settings:
        parser.add_argument('--use_mutual', type=str2bool, default=False)
        parser.add_argument('--beta', type=float, default=0.2)
        parser.add_argument('--concat_decoder_input', type=str2bool, default=True)
        parser.add_argument('--gmm', type=str2bool, default=True)

        return parser

    def init_gaussian(self):
        self._log_uniform_y = Variable(torch.log(torch.ones(1) / self.config.k))
        if self.use_gpu:
            self._log_uniform_y = self.log_uniform_y.cuda()

        mus = torch.randn(self.config.mult_k, self.config.k, self.config.latent_size)
        logvar = torch.randn(self.config.mult_k, self.config.k, self.config.latent_size)
        if torch.cuda.is_available():
            mus = mus.cuda()
            logvar = logvar.cuda()
        self._gaussian_mus = torch.nn.Parameter(mus, requires_grad=True)  # change: False
        self._gaussian_logvar = torch.nn.Parameter(logvar, requires_grad=True)  # change: False

    @property
    def gaussian_mus(self):
        return self._gaussian_mus

    @property
    def gaussian_logvar(self):
        return self._gaussian_logvar

    @property
    def log_uniform_y(self):
        return self._log_uniform_y

    def model_sel_loss(self, loss, batch_cnt):  # return albo
        if "sel_metric" in self.config and self.config.sel_metric == "elbo":
            return loss.elbo
        return self.valid_loss(loss)

    def valid_loss(self, loss, batch_cnt=None, step=None):
        # loss = Pack(nll=nll, agg_ckl=agg_ckl, mi=mi, zkl=zkl, mean_var=mean_var, PPL=ppl,
        #                            real_zkl=zkl_real, real_ckl=ckl_real, elbo=nll + ckl_real + zkl_real,
        #                            param_var=self.param_var(tgt_probs=qc))

        if batch_cnt is not None:
            step = batch_cnt

        if step is not None and 'anneal_function' in self.config:
            vae_kl_weight = kl_anneal_function(self.config.anneal_function, step,
                                               self.config.anneal_k, self.config.anneal_x0,
                                               self.config.anneal_warm_up_step if "anneal_warm_up_step" in self.config else 0,
                                               self.config.anneal_warm_up_step if "anneal_warm_up_value" in self.config else 0)
        else:
            vae_kl_weight = 1.0

        if not self.config.anneal:
            vae_kl_weight = 1.0

        mi_weight = 0.0 if self.config.use_mutual else 1.0
        total_loss = loss.nll + vae_kl_weight * loss.zkl
        return total_loss

    def reparameterization(self, mu, logvar, sample=True):
        if self.training or sample:
            std = torch.exp(0.5 * logvar)
            z = self.torch2var(torch.randn(mu.size()))
            z = z * std + mu
            return z
        else:
            return mu

    def zkl_loss(self, tgt_probs, mean, log_var, mean_prior=True):
        mean = mean.view(-1, self.config.mult_k, self.config.latent_size)
        log_var = log_var.view(-1, self.config.mult_k, self.config.latent_size)
        if mean_prior:
            tgt_probs_ = tgt_probs.unsqueeze(-1).expand(-1, -1, -1, self.config.latent_size)
            eta1 = self.gaussian_mus / torch.exp(self.gaussian_logvar)  # eta1 = \Sigma^-1 * mu
            eta2 = -0.5 * torch.pow(torch.exp(self.gaussian_logvar), -1)
            Eeta1 = torch.sum(tgt_probs_ * eta1, dim=-2)  # [batch_size, mult_k, latent_size]
            Eeta2 = torch.sum(tgt_probs_ * eta2, dim=-2)
            Emu = -0.5 * Eeta1 / Eeta2
            Evar = -0.5 * torch.pow(Eeta2, -1)
            # [batch_size, mult_k, latent_size]
            kl = 0.5 * (
                    torch.sum(log_var.exp().div(Evar), dim=-1)
                    + torch.sum((Emu - mean).pow(2) / Evar, dim=-1)
                    - mean.size(-1)
                    + torch.sum(Evar.log() - log_var, dim=-1)
            )
            # [batch_size, mult_k]
            return kl

        mu_repeat = mean.unsqueeze(-2).expand(-1, -1, self.config.k, -1)  # batch_size x k x z_dim
        logvar_repeat = log_var.unsqueeze(-2).expand(-1, -1, self.config.k, -1)
        gaussian_logvars = self.gaussian_logvar

        kl = 0.5 * (
                torch.sum(logvar_repeat.exp().div(gaussian_logvars.exp()), dim=-1)
                + torch.sum((self.gaussian_mus - mu_repeat).pow(2) / gaussian_logvars.exp(), dim=-1)
                - mean.size(-1)
                + torch.sum((gaussian_logvars - logvar_repeat), dim=-1)
        )  # batch_size x mult_k x k

        return torch.sum(kl * tgt_probs, dim=-1)  # batch_size*mult_k

    def dispersion(self, tgt_probs):
        # tgt_probs: batch_size x mult_k x k
        tgt_probs_ = tgt_probs.unsqueeze(-1).expand(-1, -1, -1, self.config.latent_size)
        eta1 = self.gaussian_mus / torch.exp(self.gaussian_logvar)  # eta1 = \Sigma^-1 * mu
        eta2 = -0.5 * torch.pow(torch.exp(self.gaussian_logvar), -1)
        Eeta1 = torch.sum(tgt_probs_ * eta1, dim=-2)  # [batch_size, mult_k, latent_size]
        Eeta2 = torch.sum(tgt_probs_ * eta2, dim=-2)
        AE = -0.25 * Eeta1 * Eeta1 / Eeta2 - 0.5 * torch.log(-2 * Eeta2)  # [batch_size, mult_k, latent_size]
        AE = torch.mean(torch.sum(AE, dim=(-1, -2)))

        EA = torch.sum(-0.25 * eta1 * eta1 / eta2 - 0.5 * torch.log(-2 * eta2), dim=-1)  # [mult_k, k]
        EA = torch.mean(torch.sum(tgt_probs * EA, dim=(-1, -2)))
        return EA - AE

    def param_var(self, tgt_probs):
        # Weighted variance of natural parameters
        # tgt_probs: batch_size x mult_k x k
        tgt_probs_ = tgt_probs.unsqueeze(-1).expand(-1, -1, -1, self.config.latent_size)
        eta1 = self.gaussian_mus / torch.exp(self.gaussian_logvar)  # eta1 = \Sigma^-1 * mu
        eta2 = -0.5 * torch.pow(torch.exp(self.gaussian_logvar), -1)

        var_eta1 = torch.sum(tgt_probs_ * (eta1 * eta1), dim=-2) - torch.sum(tgt_probs_ * eta1, dim=-2).pow(2)
        var_eta2 = torch.sum(tgt_probs_ * (eta2 * eta2), dim=-2) - torch.sum(tgt_probs_ * eta2, dim=-2).pow(2)

        return torch.sum(var_eta1 + var_eta2) / tgt_probs.size(0)

    def _get_pzc(self, sample_z):
        # sample_z: [batch_size, latent_size * multi_k]
        # Prior: [multi_k, k, latent_size]
        bsz = sample_z.size(0)
        multi_k, k, ls = self.gaussian_mus.size()
        gaussian_mus = self.gaussian_mus.unsqueeze(0).expand(bsz, multi_k, k, ls)
        gaussian_logvar = self.gaussian_logvar.unsqueeze(0).expand(bsz, multi_k, k, ls)
        sample_z = sample_z.view(-1, multi_k, 1, ls).expand(bsz, multi_k, k, ls)
        log_pz = - 0.5 * (sample_z - gaussian_mus) * (sample_z - gaussian_mus) / \
                 torch.exp(gaussian_logvar) - 0.5 * math.log(math.pi * 2) - 0.5 * gaussian_logvar
        return torch.sum(log_pz, dim=-1)


    def forward(self, data_feed, mode, gen_type='greedy', sample_n=1, return_latent=False):
        posterior_sample_n = self.posterior_sample_n if self.training else 1

        if type(data_feed) is tuple:
            data_feed = data_feed[0]
        batch_size = len(data_feed['output_lens'])
        out_utts = self.np2var(data_feed['outputs'], LONG)

        # output encoder
        output_embedding = self.embedding(out_utts)
        x_outs, x_last = self.x_encoder(output_embedding)
        if type(x_last) is tuple:
            x_last = x_last[0].view(self.num_layer_enc, 1 + int(self.bi_enc_cell), -1, self.enc_cell_size)[-1]
            x_last = x_last.transpose(0, 1).contiguous().view(-1, self.enc_out_size)
        else:
            x_last = x_last.view(self.num_layer_enc, 1 + int(self.bi_enc_cell), -1, self.enc_cell_size)[-1]
            x_last = x_last.transpose(0, 1).contiguous().view(-1,
                                                              self.enc_out_size)

        # q(z|x)
        qz_mean = self.q_y_mean(x_last)  # batch x (latent_size*mult_k)
        qz_logvar = self.q_y_logvar(x_last)
        sample_z = self.reparameterization(qz_mean.repeat(posterior_sample_n, 1),
                                           qz_logvar.repeat(posterior_sample_n, 1),
                                           sample=gen_type != "greedy" or mode != GEN)  # batch x (latent_size*mult_k)
        # q(c|x)
        qc_logits = self.post_c(x_last).view(-1, self.config.k)
        log_qc = F.log_softmax(qc_logits, -1)  # [batch*mult_k, k]
        sample_c, c_ids = self.cat_connector(qc_logits, 1.0, self.use_gpu, hard=not self.training, return_max_id=True)
        # sample_c: [batch*mult_k, k], c_ids: [batch*mult_k, 1]
        # sample_c = sample_c.view(-1, self.config.mult_k * self.config.k)
        c_ids = c_ids.view(-1, self.config.mult_k)

        # Prepare for decoding
        dec_init_state = self.dec_init_connector(sample_z)
        labels = out_utts[:, 1:].contiguous()
        dec_inputs = out_utts[:, 0:-1]
        if self.config.word_dropout_rate > 0:
            # randomly replace decoder input with <unk>
            prob = torch.rand(dec_inputs.size())
            prob[(dec_inputs.data - self.go_id) * (dec_inputs.data - self.pad_id) * (
                        dec_inputs.data - self.eos_id) == 0] = 1
            dec_inputs_copy = dec_inputs.clone()
            dec_inputs_copy[prob < self.config.word_dropout_rate] = self.unk_id
            dec_inputs = dec_inputs_copy

        # decode
        dec_outs, dec_last, dec_ctx = self.decoder(batch_size * posterior_sample_n,
                                                   dec_inputs.repeat(posterior_sample_n, 1),
                                                   dec_init_state,
                                                   mode=mode, gen_type=gen_type,
                                                   beam_size=self.beam_size,
                                                   latent_variable=sample_z if self.concat_decoder_input else None)

        # compute loss or return results
        if mode == GEN:
            return dec_ctx, labels
        else:
            # RNN reconstruction
            nll = self.nll_loss(dec_outs, labels.repeat(posterior_sample_n, 1))
            ppl = self.ppl(dec_outs, labels.repeat(posterior_sample_n, 1))
            # Regularization terms
            # ZKL:
            log_qz = - 0.5 * (sample_z - qz_mean.repeat(posterior_sample_n, 1)) \
                     * (sample_z - qz_mean.repeat(posterior_sample_n, 1)) / torch.exp(qz_logvar.repeat(posterior_sample_n, 1)) \
                     - 0.5 * qz_logvar.repeat(posterior_sample_n, 1) - 0.5 * math.log(math.pi * 2)
            log_qz = torch.sum(log_qz, dim=-1)
            log_pzc = self._get_pzc(sample_z) # [batch_size x multi_k x k]
            log_pz = torch.sum(torch.log(torch.mean(torch.exp(log_pzc), dim=-1) + 1e-15), dim=-1)
            zkl = torch.mean(log_qz - log_pz)
            # qc = q(z|x) * p(c|z)
            log_qc = F.log_softmax(log_pzc, dim=-1)
            qc = torch.exp(log_qc.view(-1, self.config.mult_k, self.config.k))

            dispersion = self.dispersion(qc)
            # MI term
            avg_log_qc = torch.log(torch.mean(qc, dim=0) + 1e-15)
            mi = - torch.sum(torch.exp(avg_log_qc) * avg_log_qc) + torch.sum(torch.exp(log_qc) * log_qc) / log_qc.size(0)

            results = Pack(nll=nll, mi=mi, zkl=zkl, dispersion=dispersion, PPL=ppl,
                           elbo=nll + zkl,
                           param_var=self.param_var(tgt_probs=qc))

            if return_latent:
                results['log_qy'] = log_qc
                results['dec_init_state'] = dec_init_state
                results['y_ids'] = c_ids
                results['z'] = sample_z

            return results

    def sampling_for_likelihood(self, batch_size, data_feed, sample_num, sample_type="LL",
                                ):
        # Importance sampling for estimating the log-likelihood
        assert sample_type in ("LL", "logLL")

        if type(data_feed) is tuple:
            data_feed = data_feed[0]
        batch_size = len(data_feed['output_lens'])
        out_utts = self.np2var(data_feed['outputs'], LONG)  # batch_size * seq_len
        out_utts = out_utts.repeat(sample_num, 1)

        labels = out_utts[:, 1:].contiguous()
        dec_inputs = out_utts[:, 0:-1]

        output_embedding = self.embedding(out_utts)
        x_outs, x_last = self.x_encoder(output_embedding)
        if type(x_last) is tuple:
            x_last = x_last[0].view(self.num_layer_enc, 1 + int(self.bi_enc_cell), -1, self.enc_cell_size)[-1]
            x_last = x_last.transpose(0, 1).contiguous().view(-1, self.enc_out_size)
        else:
            x_last = x_last.view(self.num_layer_enc, 1 + int(self.bi_enc_cell), -1, self.enc_cell_size)[-1]
            x_last = x_last.transpose(0, 1).contiguous().view(-1,
                                                              self.enc_out_size)

        # q(z|x)
        qz_mean = self.q_y_mean(x_last)  # batch x (latent_size*mult_k)
        qz_logvar = self.q_y_logvar(x_last)
        sample_z = self.reparameterization(qz_mean, qz_logvar, sample=True)

        log_qzx = torch.sum(
            - (sample_z - qz_mean) * (sample_z - qz_mean) / (
                        2 * torch.exp(qz_logvar)) - 0.5 * qz_logvar - 0.5 * math.log(
                math.pi * 2),
            dim=-1)
        sample_z_repeat = sample_z.view(-1, self.config.mult_k, 1, self.config.latent_size).repeat(1, 1, self.config.k,
                                                                                                   1)
        log_pzc = torch.sum(
            - (sample_z_repeat - self.gaussian_mus) * (sample_z_repeat - self.gaussian_mus) / (
                        2 * torch.exp(self.gaussian_logvar))
            - 0.5 * self.gaussian_logvar - 0.5 * math.log(math.pi * 2),
            dim=-1)  # [batch_size, mult_k, k]
        log_pz = torch.log(torch.mean(torch.exp(log_pzc.double()), dim=-1))  #
        log_pz = torch.sum(log_pz, dim=-1)

        # Calculate p(x|z)
        dec_init_state = self.dec_init_connector(sample_z)
        dec_outs, dec_last, outputs = self.decoder(sample_z.size(0),
                                                   dec_inputs, dec_init_state,
                                                   mode=TEACH_FORCE,
                                                   gen_type=self.config.gen_type,
                                                   beam_size=self.config.beam_size,
                                                   latent_variable=sample_z if self.concat_decoder_input else None)

        nll = F.nll_loss(dec_outs.view(-1, dec_outs.size(-1)), labels.view(-1), reduction="none").view(out_utts.size(0),
                                                                                                       -1)
        nll = torch.sum(nll, dim=-1)
        if sample_type == "logLL":
            return (-nll.double() + log_pz - log_qzx.double()).view(-1, sample_num)
        else:
            ll = torch.exp(-nll.double() + log_pz - log_qzx.double())  # exp ( log (p(z)p(x|z) / q(z|x)) )
            ll = ll.view(-1, sample_num)
        return ll

    def sampling(self, batch_size):
        sample_c = torch.randint(0, self.config.k, [batch_size, self.config.mult_k], dtype=torch.long).cuda()
        index = (self.torch2var(torch.arange(self.config.mult_k) * self.config.k) + sample_c).view(-1)
        mean = self.gaussian_mus.view(-1, self.config.latent_size)[index].squeeze()
        sigma = torch.exp(self.gaussian_logvar * 0.5).view(-1, self.config.latent_size)[index].squeeze()
        zs = self.reparameterization(mean, 2 * torch.log(torch.abs(sigma) + 1e-15), sample=True)
        zs = zs.view(-1, self.config.mult_k * self.config.latent_size)
        dec_init_state = self.dec_init_connector(zs)
        _, _, outputs = self.decoder(zs.size(0),
                                     None, dec_init_state,
                                     mode=GEN, gen_type="greedy",
                                     beam_size=self.config.beam_size,
                                     latent_variable=zs if self.concat_decoder_input else None)
        return outputs

class VAE(BaseModel):
    def __init__(self, corpus, config):
        super(VAE, self).__init__(config)
        self.vocab = corpus.vocab
        self.rev_vocab = corpus.rev_vocab
        self.vocab_size = len(self.vocab)
        self.embed_size = config.embed_size
        self.max_utt_len = config.max_utt_len
        self.go_id = self.rev_vocab[BOS]
        self.eos_id = self.rev_vocab[EOS]
        self.unk_id = self.rev_vocab[UNK]
        self.pad_id = self.rev_vocab[PAD]
        self.num_layer_enc = config.num_layer_enc
        self.num_layer_dec = config.num_layer_dec
        self.dropout = config.dropout
        self.enc_cell_size = config.enc_cell_size
        self.dec_cell_size = config.dec_cell_size
        self.rnn_cell = config.rnn_cell
        self.max_dec_len = config.max_dec_len
        self.use_attn = config.use_attn
        self.beam_size = config.beam_size
        self.utt_type = config.utt_type
        self.bi_enc_cell = config.bi_enc_cell
        self.attn_type = config.attn_type
        self.enc_out_size = self.enc_cell_size * 2 if self.bi_enc_cell else self.enc_cell_size
        self.posterior_sample_n = config.post_sample_num if "post_sample_num" in config else 1
        self.concat_decoder_input = config.concat_decoder_input if "concat_decoder_input" in config else False
        self.use_kl = getattr(config, "use_kl", True)

        # build model here
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size,
                                      padding_idx=self.rev_vocab[PAD])

        self.x_encoder = EncoderRNN(self.embed_size, self.enc_cell_size,
                                    dropout_p=self.dropout,
                                    rnn_cell=self.rnn_cell,
                                    variable_lengths=self.config.fix_batch,
                                    bidirection=self.bi_enc_cell,
                                    n_layers=self.num_layer_enc
                                )

        self.q_z_mean = nn.Linear(self.enc_out_size, config.latent_size)
        self.q_z_logvar = nn.Linear(self.enc_out_size, config.latent_size)

        self.cat_connector = nn_lib.GumbelConnector()
        self.dec_init_connector = nn_lib.LinearConnector(config.latent_size,
                                                         self.dec_cell_size,
                                                         self.rnn_cell == 'lstm',
                                                         has_bias=False)

        self.decoder = DecoderRNN(self.vocab_size, self.max_dec_len,
                                  self.embed_size + self.config.latent_size if self.concat_decoder_input else self.embed_size,
                                  self.dec_cell_size,
                                  self.go_id, self.eos_id, self.unk_id,
                                  n_layers=self.num_layer_dec, rnn_cell=self.rnn_cell,
                                  input_dropout_p=self.dropout,
                                  dropout_p=self.dropout,
                                  use_attention=self.use_attn,
                                  attn_size=self.enc_cell_size,
                                  attn_mode=self.attn_type,
                                  use_gpu=self.use_gpu,
                                  embedding=self.embedding,
                                  softmax_temperature=self.config.softmax_temperature if "softmax_temperature" in self.config else 1.0)

        self.nll_loss = criterions.NLLEntropy(self.rev_vocab[PAD], self.config)
        self.ppl = criterions.Perplexity(self.rev_vocab[PAD], self.config)
        self.cat_kl_loss = criterions.CatKLLoss()
        self.cross_ent_loss = criterions.CrossEntropyoss()
        self.entropy_loss = criterions.Entropy()

        if 'bow_loss' in self.config and self.config.bow_loss:
            self.bow_mlp = nn.Linear(config.latent_size, self.vocab_size)
            self.bow_loss = True
            self.bow_entropy = criterions.BowEntropy(self.rev_vocab[PAD], self.config)
        else:
            self.bow_loss = False

        self.kl_w = 0.0

        self.return_latent_key = ("dec_init_state", "qz_mean", "qz_logvar", "q_z")

    @staticmethod
    def add_args(parser):
        from dgmvae.utils import str2bool
        # Latent variable:
        parser.add_argument('--latent_size', type=int, default=40, help="The latent size of continuous latent variable.")

        # Network setting:
        parser.add_argument('--rnn_cell', type=str, default='gru')
        parser.add_argument('--embed_size', type=int, default=512)
        parser.add_argument('--utt_type', type=str, default='rnn')
        parser.add_argument('--enc_cell_size', type=int, default=512)
        parser.add_argument('--dec_cell_size', type=int, default=512)
        parser.add_argument('--bi_enc_cell', type=str2bool, default=True)
        parser.add_argument('--num_layer_enc', type=int, default=1)
        parser.add_argument('--num_layer_dec', type=int, default=1)
        parser.add_argument('--use_attn', type=str2bool, default=False)
        parser.add_argument('--attn_type', type=str, default='cat')
        parser.add_argument('--tie_output_embed', type=str2bool, default=True)
        parser.add_argument('--max_utt_len', type=int, default=40)
        parser.add_argument('--max_dec_len', type=int, default=40)
        parser.add_argument('--max_vocab_cnt', type=int, default=10000)

        parser.add_argument('--use_kl', type=str2bool, default=True, help="use_kl=False: AE; use_kl=True, VAE.")
        parser.add_argument('--bow_loss', type=str2bool, default=False, help="adding bow loss to objective.")
        parser.add_argument('--concat_decoder_input', type=str2bool, default=True)
        parser.add_argument('--gmm', type=str2bool, default=False)

        return parser

    def valid_loss(self, loss, batch_cnt=None, step = None):
        if batch_cnt is not None:
            step = batch_cnt

        if step is not None and 'anneal_function' in self.config:
            vae_kl_weight = kl_anneal_function(self.config.anneal_function, step,
                                               self.config.anneal_k, self.config.anneal_x0)
        else:
            vae_kl_weight = 1.0

        if not self.use_kl:
            loss.KL_loss = 0.0
        total_loss = loss.nll + vae_kl_weight * loss.KL_loss

        if self.bow_loss and self.training:
            total_loss += loss.bow_loss

        return total_loss

    def model_sel_loss(self, loss, batch_cnt): # return albo
        if not self.use_kl:
            return loss.nll
        return loss.ELBO

    def reparameterization(self, mu, logvar, batch=False, sample=True):
        if not self.use_kl:
            sample = False
        if self.training or sample:
            std = torch.exp(0.5 * logvar)
            z = self.torch2var(torch.randn(mu.size()))
            z = z * std + mu
            return z
        else:
            return mu

    def forward(self, data_feed, mode, gen_type='greedy', sample_n=1, return_latent=False):
        posterior_sample_n = self.posterior_sample_n if self.training else 1

        if type(data_feed) is tuple:
            data_feed = data_feed[0]
        batch_size = len(data_feed['output_lens'])
        out_utts = self.np2var(data_feed['outputs'], LONG)

        # output encoder
        output_embedding = self.embedding(out_utts)
        x_outs, x_last = self.x_encoder(output_embedding)
        if type(x_last) is tuple:
            x_last = x_last[0].view(self.num_layer_enc, 1 + int(self.bi_enc_cell), -1, self.enc_cell_size)[-1]
            x_last = x_last.transpose(0, 1).contiguous().view(-1, self.enc_out_size)
        else:
            x_last = x_last.view(self.num_layer_enc, 1 + int(self.bi_enc_cell), -1, self.enc_cell_size)[-1]
            x_last = x_last.transpose(0, 1).contiguous().view(-1,
                                                              self.enc_out_size)

        # posterior network
        qz_mean = self.q_z_mean(x_last)
        qz_logvar = self.q_z_logvar(x_last)
        q_z = self.reparameterization(qz_mean.repeat(posterior_sample_n, 1),
                                      qz_logvar.repeat(posterior_sample_n, 1), batch=True,
                                      sample=gen_type != "greedy" or mode != GEN)  # batch x (latent_size*mult_k)

        # map sample to initial state of decoder
        dec_init_state = self.dec_init_connector(q_z)

        # get decoder inputs
        labels = out_utts[:, 1:].contiguous()
        dec_inputs = out_utts[:, 0:-1]

        if self.config.word_dropout_rate > 0:
            # randomly replace decoder input with <unk>
            prob = torch.rand(dec_inputs.size())
            prob[(dec_inputs.data - self.go_id) * (dec_inputs.data - self.pad_id) == 0] = 1
            decoder_input_sequence = dec_inputs.clone()
            decoder_input_sequence[prob < self.config.word_dropout_rate] = self.unk_id
            # input_embedding = self.embedding(decoder_input_sequence)
            dec_inputs = decoder_input_sequence

        # decode
        dec_outs, dec_last, dec_ctx = self.decoder(batch_size * posterior_sample_n,
                                                   dec_inputs.repeat(posterior_sample_n, 1),
                                                   dec_init_state,
                                                   mode=mode, gen_type=gen_type,
                                                   beam_size=self.beam_size,
                                                   latent_variable=q_z if self.concat_decoder_input else None)
        # compute loss or return results
        if mode == GEN:
            return dec_ctx, labels
        else:
            # RNN reconstruction
            nll = self.nll_loss(dec_outs, labels.repeat(posterior_sample_n, 1))
            ppl = self.ppl(dec_outs, labels.repeat(posterior_sample_n, 1))
            KL_loss = -0.5 * torch.mean(torch.sum((1 + qz_logvar - qz_mean.pow(2) - qz_logvar.exp()), dim=1))

            if not self.use_kl:
                KL_loss = torch.zeros([]).cuda()

            if self.bow_loss:
                bow_logits = self.bow_mlp(q_z)
                bow_loss = self.bow_entropy(F.log_softmax(bow_logits), labels)
            else:
                bow_loss = torch.zeros([]).cuda()

            results = Pack(nll=nll, KL_loss=KL_loss, ELBO=nll+KL_loss, PPL=ppl, bow_loss=bow_loss)

            if return_latent:
                for key in self.return_latent_key:
                    results[key] = eval(key)
            return results

    def sampling_for_likelihood(self, batch_size, data_feed, sample_num, sample_type="LL"):
        # Importance sampling...
        assert sample_type in ("LL", "logLL")

        if type(data_feed) is tuple:
            data_feed = data_feed[0]
        batch_size = len(data_feed['output_lens'])
        out_utts = self.np2var(data_feed['outputs'], LONG)  # batch_size * seq_len
        out_utts = out_utts.repeat(sample_num, 1)
        labels = out_utts[:, 1:].contiguous()
        dec_inputs = out_utts[:, 0:-1]

        output_embedding = self.embedding(out_utts)
        x_outs, x_last = self.x_encoder(output_embedding)

        if type(x_last) is tuple:
            x_last = x_last[0].view(self.num_layer_enc, 1 + int(self.bi_enc_cell), -1, self.enc_cell_size)[-1]
            x_last = x_last.transpose(0, 1).contiguous().view(-1, self.enc_out_size)
        else:
            x_last = x_last.view(self.num_layer_enc, 1 + int(self.bi_enc_cell), -1, self.enc_cell_size)[-1]
            x_last = x_last.transpose(0, 1).contiguous().view(-1,
                                                              self.enc_out_size)

        qz_mean = self.q_z_mean(x_last)  # [batch_size * sample_num, latent_size]
        qz_logvar = self.q_z_logvar(x_last)
        q_z = self.reparameterization(qz_mean, qz_logvar, batch=True, sample=True)

        log_qzx = torch.sum(
            - (q_z - qz_mean) * (q_z - qz_mean) / (2 * torch.exp(qz_logvar)) -0.5 * qz_logvar - 0.5 * math.log(math.pi * 2),
            dim=-1)

        dec_init_state = self.dec_init_connector(q_z)
        dec_outs, dec_last, outputs = self.decoder(q_z.size(0),
                                                   dec_inputs, dec_init_state,
                                                   mode=TEACH_FORCE,
                                                   gen_type=self.config.gen_type,
                                                   beam_size=self.config.beam_size,
                                                   latent_variable=q_z if self.concat_decoder_input else None)

        nll = F.nll_loss(dec_outs.view(-1, dec_outs.size(-1)), labels.view(-1), reduction="none").view(out_utts.size(0), -1)
        nll = torch.sum(nll, dim=-1)

        log_pz = torch.sum(- 0.5 * q_z * q_z - 0.5 * math.log(math.pi * 2), dim=-1) # [batch_size * sample_num, ]

        ll = torch.exp(-nll.double() + log_pz.double() - log_qzx.double())  # log (p(z)p(x|z) / q(z|x))

        if sample_type == "logLL":
            return (-nll.double() + log_pz.double() - log_qzx.double()).view(-1, sample_num)
        else:
            ll = ll.view(-1, sample_num)
        return ll

    def sampling(self, batch_size):
        zs = self.torch2var(torch.randn(batch_size, self.config.latent_size))
        dec_init_state = self.dec_init_connector(zs)

        dec_outs, dec_last, outputs = self.decoder(zs.size(0),
                                      None, dec_init_state,
                                      mode=GEN, gen_type="greedy",
                                      beam_size=self.config.beam_size,
                                      latent_variable=zs)

        return outputs

class RNNLM(BaseModel):
    def __init__(self, corpus, config):
        super(RNNLM, self).__init__(config)
        self.vocab = corpus.vocab
        self.rev_vocab = corpus.rev_vocab
        self.vocab_size = len(self.vocab)
        self.embed_size = config.embed_size
        self.max_utt_len = config.max_utt_len
        self.go_id = self.rev_vocab[BOS]
        self.eos_id = self.rev_vocab[EOS]
        self.unk_id = self.rev_vocab[UNK]
        self.num_layer = config.num_layer
        self.dropout = config.dropout
        self.dec_cell_size = config.dec_cell_size
        self.rnn_cell = config.rnn_cell
        self.max_dec_len = config.max_dec_len
        self.beam_size = config.beam_size
        self.utt_type = config.utt_type

        # build model here
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size,
                                      padding_idx=self.rev_vocab[PAD])

        self.decoder = DecoderRNN(self.vocab_size, self.max_dec_len,
                                  self.embed_size, self.dec_cell_size,
                                  self.go_id, self.eos_id, self.unk_id,
                                  n_layers=config.num_layer, rnn_cell=self.rnn_cell,
                                  input_dropout_p=self.dropout,
                                  dropout_p=self.dropout,
                                  use_attention=False,
                                  # attn_size=self.enc_cell_size,
                                  # attn_mode='cat',
                                  use_gpu=self.use_gpu,
                                  embedding=self.embedding)

        self.nll_loss = criterions.NLLEntropy(self.rev_vocab[PAD], self.config)
        self.ppl = criterions.Perplexity(self.rev_vocab[PAD], self.config)
        self.cat_kl_loss = criterions.CatKLLoss()
        self.cross_ent_loss = criterions.CrossEntropyoss()
        self.entropy_loss = criterions.Entropy()

        # self.kl_w = 0.0

        for para in self.parameters():
            nn.init.uniform_(para.data, -0.1, 0.1)

        # self.return_latent_key = ("dec_init_state", "qy_mean", "qy_logvar", "q_z")

    @staticmethod
    def add_args(parser):
        from dgmvae.utils import str2bool
        # Network setting:
        parser.add_argument('--rnn_cell', type=str, default='gru')
        parser.add_argument('--embed_size', type=int, default=512)
        parser.add_argument('--utt_type', type=str, default='rnn')
        parser.add_argument('--dec_cell_size', type=int, default=512)
        parser.add_argument('--num_layer', type=int, default=1)
        parser.add_argument('--tie_output_embed', type=str2bool, default=True)
        parser.add_argument('--max_dec_len', type=int, default=40)
        parser.add_argument('--max_utt_len', type=int, default=40)
        parser.add_argument('--max_vocab_cnt', type=int, default=10000)
        return parser

    def valid_loss(self, loss, batch_cnt=None, step = None):
        return loss.nll

    def model_sel_loss(self, loss, batch_cnt):
        return loss.nll

    def reparameterization(self, mu, logvar, batch=False, sample=False):
        if 'use_KL' in self.config and not self.config.use_KL:
            sample = False
        if self.training or sample:
            std = torch.exp(0.5 * logvar)
            z = self.torch2var(torch.randn(mu.size()))
            z = z * std + mu
            return z
        else:
            return mu

    def forward(self, data_feed, mode, gen_type='greedy', sample_n=1, return_latent=False):
        if type(data_feed) is tuple:
            data_feed = data_feed[0]
        batch_size = len(data_feed['output_lens'])
        out_utts = self.np2var(data_feed['outputs'], LONG)

        # map sample to initial state of decoder
        # dec_init_state = self.dec_init_connector(q_z)

        # get decoder inputs
        labels = out_utts[:, 1:].contiguous()
        dec_inputs = out_utts[:, 0:-1]

        # decode
        dec_outs, dec_last, dec_ctx = self.decoder(batch_size,
                                                   dec_inputs, None,  # dec_init_state
                                                   mode=mode, gen_type=gen_type,
                                                   beam_size=self.beam_size)
        # compute loss or return results
        if mode == GEN:
            return dec_ctx, labels
        else:
            # RNN reconstruction
            nll = self.nll_loss(dec_outs, labels)
            ppl = self.ppl(dec_outs, labels)

            results = Pack(nll=nll, PPL=ppl)

            if return_latent:
                for key in self.return_latent_key:
                    results[key] = eval(key)
            return results

    def sampling(self, batch_size):
        _, _, outputs = self.decoder(batch_size,
                                     None, None,  # dec_init_state
                                     mode=GEN, gen_type="sample",
                                     beam_size=self.beam_size)

        return outputs
